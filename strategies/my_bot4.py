# Name: Your Name
# College: Your College
# Roll Number: Your Roll Number

import math
import random
from typing import Dict, List, Tuple, Optional, Any, FrozenSet
from collections import defaultdict

class Bot:
    """
    Optimal Bayesian Bot for QuantStorm 2026.
    Uses exact posterior distribution of S, optimal spread selection,
    and heuristic power valuation with TE salvage consideration.
    """
    name = "OptimalBayesianBot"

    # ---------- Required methods ----------

    def reset(self, seat: int, config: Any, seed: int) -> None:
        """Initialize per‑deal state."""
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)

        # State carried across rounds
        self.round = 0
        self.my_revealed_sum = 0          # sum of my revealed coins so far
        self.foresight_sum = 0            # sum of opponent coins seen via FORESIGHT (this round)
        self.foresight_count = 0          # number of coins seen via FORESIGHT

        self.te_mine = config.TE_BUDGET
        self.powers_mine = set()
        self.powers_theirs = set()
        self.auction_log = []
        self.contracts = []

        # Shift powers (TRICK_ROOM, STEALTH_ROCK)
        self.shift_mine = 0
        self.shift_theirs = 0
        self.stealth_rock_mine = False    # whether I hold persistent shift
        self.stealth_rock_theirs = False

        # SUBSTITUTE active this round
        self.substitute_mine = False

        # TRANSFORM
        self.transform_used = False

        # Cache for distributions of sum of n i.i.d. ±1 coins
        self.dist_cache = {}

    def bid(self, obs: Any, offered: List[str]) -> Dict[str, int]:
        """Submit blind TE bids for the offered power."""
        self._update_state(obs)

        bids = {}
        for power in offered:
            value = self._estimate_power_value(power, obs)
            if value <= 0.1:
                continue
            # Bid a fraction of value, with TE salvage opportunity cost.
            # Effective cost = bid * (1 + TE_SALVAGE) because spending TE loses salvage.
            # Optimal bid in first‑price auction with unknown opponent: shade to ~0.6 * value.
            bid = int(0.6 * value / (1 + self.config.TE_SALVAGE))  # convert to TE
            if bid > 0:
                bids[power] = min(bid, obs.te_mine)  # cap to remaining TE

        # If total bids exceed TE, the engine will zero all bids (per rules).
        # We avoid that by trimming if necessary.
        total = sum(bids.values())
        if total > obs.te_mine:
            # Scale down proportionally
            scale = obs.te_mine / total
            for p in bids:
                bids[p] = int(bids[p] * scale)
            # Ensure we don't overshoot due to rounding
            while sum(bids.values()) > obs.te_mine:
                # reduce largest bid by 1
                max_p = max(bids, key=bids.get)
                bids[max_p] -= 1
                if bids[max_p] == 0:
                    del bids[max_p]
        return bids

    def quote(self, obs: Any) -> Tuple[int, int]:
        """Maker's opening quote: tight spread around our posterior mean."""
        self._update_state(obs)
        mean = self._posterior_mean(obs)
        floor = obs.final_cap
        # We choose the minimum allowed width to reduce width premium and force action.
        # Center the spread on our mean.
        bid = int(math.floor(mean - floor / 2))
        ask = bid + floor
        # Clamp to reasonable range (S is between -40 and 40, but we allow a bit more)
        bid = max(-40, min(40 - floor, bid))
        ask = bid + floor
        return (bid, ask)

    def respond(self, obs: Any, quote: Tuple[int, int], turn: int):
        """Taker's response: accept if price is favourable, else counter."""
        self._update_state(obs)
        bid, ask = quote
        mean = self._posterior_mean(obs)
        # We have shift power? (TRICK_ROOM or STEALTH_ROCK)
        my_shift = self.shift_mine
        their_shift = self.shift_theirs
        # If we have shift, we can profit from forced fills.
        # On the last turn, countering forces midpoint and we pay 2 ticks.
        if turn == obs.n_turns:
            # Last turn: we cannot counter without forcing and paying fee.
            # Decide: accept buy, accept sell, or force (counter) if shift makes it worthwhile.
            # Forced price = midpoint + (my_shift - their_shift)  (since shift moves in our favour)
            midpoint = (bid + ask) // 2
            forced_price = midpoint + (my_shift - their_shift)
            # If we accept buy at ask, profit = mean - ask (if we are long)
            # If we accept sell at bid, profit = bid - mean (if we are short)
            # We choose the action with highest expected profit.
            profit_buy = mean - ask
            profit_sell = bid - mean
            profit_force = mean - forced_price if (forced_price < mean) else forced_price - mean  # we can choose side? Actually forced price is the execution price; if we are long (buy) profit = S - price, but we don't know S. We use our mean as expectation.
            # We prefer positive profit; if all negative, choose least loss.
            best_action = None
            best_profit = -1e9
            # Evaluate buy
            if profit_buy > best_profit:
                best_profit = profit_buy
                best_action = "ACCEPT_BUY"
            if profit_sell > best_profit:
                best_profit = profit_sell
                best_action = "ACCEPT_SELL"
            # Forcing: we pay 2 ticks, but we get shift advantage.
            # Net profit if we force: we expect to buy at forced_price if we are long? Actually we don't choose side; the last quoter is short. So if we force, we are the short seat (sell). So our profit = price - S? Wait: if we are the last quoter (counter on turn 6), we are the short seat (seller). So we sell at forced_price, profit = forced_price - S. Expected profit = forced_price - mean.
            if my_shift != their_shift:
                # If we have shift advantage, forced price may be better.
                profit_force_sell = forced_price - mean  # we are short
                if profit_force_sell > best_profit - 2.0:  # subtract forcing fee
                    best_profit = profit_force_sell - 2.0
                    best_action = ("COUNTER", bid, ask)  # counter with same quote to force
            # If we decide to force, we must return a counter that shrinks? Actually we can return the same range; it will be clamped to force midpoint.
            if best_action is None:
                # Default: accept buy if ask < mean, else accept sell
                if ask < mean:
                    return "ACCEPT_BUY"
                else:
                    return "ACCEPT_SELL"
            return best_action
        else:
            # Not last turn: we can counter.
            # We want to shrink the spread towards our mean, reducing width by MIN_REDUCTION.
            # If we think the ask is too high, we might accept buy if ask < mean - margin.
            # If we think the bid is too low, accept sell if bid > mean + margin.
            margin = 0.5  # small edge
            if ask < mean - margin:
                return "ACCEPT_BUY"
            if bid > mean + margin:
                return "ACCEPT_SELL"
            # Otherwise counter: move the spread inward towards mean.
            new_bid = bid
            new_ask = ask
            # Reduce width by at least MIN_REDUCTION
            current_width = ask - bid
            min_reduction = self.config.MIN_REDUCTION
            if current_width - min_reduction < obs.final_cap:
                # cannot shrink below floor; we'll keep floor width
                target_width = obs.final_cap
            else:
                target_width = current_width - min_reduction
            # Center the new spread on our mean, but within current bounds.
            mid = (bid + ask) / 2.0
            # Move midpoint towards mean, but not beyond current range.
            new_mid = mid + 0.5 * (mean - mid)  # move halfway towards mean
            # Clamp new_mid to [bid + target_width/2, ask - target_width/2]
            lower_bound = bid + target_width / 2
            upper_bound = ask - target_width / 2
            if lower_bound > upper_bound:
                lower_bound = upper_bound = (bid + ask) / 2.0
            new_mid = max(lower_bound, min(upper_bound, new_mid))
            new_bid = int(math.floor(new_mid - target_width / 2))
            new_ask = new_bid + target_width
            # Ensure within current range
            new_bid = max(bid, min(ask - target_width, new_bid))
            new_ask = new_bid + target_width
            # Ensure we actually reduced width
            if new_ask - new_bid >= current_width:
                # If we can't shrink, accept the better side
                if ask <= mean:
                    return "ACCEPT_BUY"
                elif bid >= mean:
                    return "ACCEPT_SELL"
                else:
                    return "ACCEPT_BUY"  # default
            return ("COUNTER", new_bid, new_ask)

    def use_transform(self, obs: Any) -> bool:
        """Swap hands if our revealed sum is negative (we expect to have a bad hand)."""
        self._update_state(obs)
        # If our revealed sum is negative, we likely have more -1s than opponent's average.
        # Swap if my revealed sum < 0 and we haven't used it.
        if not self.transform_used and obs.my_revealed and sum(obs.my_revealed) < 0:
            self.transform_used = True
            return True
        self.transform_used = True
        return False

    # ---------- Internal helpers ----------

    def _update_state(self, obs: Any) -> None:
        """Synchronise internal state from the current Obs object."""
        self.round = obs.round
        self.my_revealed_sum = sum(obs.my_revealed)
        self.te_mine = obs.te_mine
        self.powers_mine = set(obs.powers_mine)
        self.powers_theirs = set(obs.powers_theirs)
        self.auction_log = list(obs.auction_log)
        self.contracts = list(obs.contracts)
        # Update shift powers
        self.shift_mine = 0
        self.shift_theirs = 0
        if "TRICK_ROOM" in self.powers_mine:
            self.shift_mine += self.config.TRICK_ROOM_MAGNITUDE
        if "TRICK_ROOM" in self.powers_theirs:
            self.shift_theirs += self.config.TRICK_ROOM_MAGNITUDE
        if "STEALTH_ROCK" in self.powers_mine:
            self.shift_mine += self.config.STEALTH_ROCK_MAGNITUDE
            self.stealth_rock_mine = True
        if "STEALTH_ROCK" in self.powers_theirs:
            self.shift_theirs += self.config.STEALTH_ROCK_MAGNITUDE
            self.stealth_rock_theirs = True
        # SUBSTITUTE
        self.substitute_mine = "SUBSTITUTE" in self.powers_mine
        # FORESIGHT
        self.foresight_sum = sum(obs.foresight) if obs.foresight else 0
        self.foresight_count = len(obs.foresight)

    def _get_dist(self, n: int) -> Dict[int, float]:
        """Return probability mass function of sum of n i.i.d. ±1 coins."""
        if n in self.dist_cache:
            return self.dist_cache[n]
        dist = {}
        # sum = 2*k - n, where k ~ Binomial(n, 0.5)
        for k in range(n + 1):
            s = 2 * k - n
            prob = math.comb(n, k) / (1 << n)
            dist[s] = prob
        self.dist_cache[n] = dist
        return dist

    def _posterior_distribution(self, obs: Any) -> Dict[int, float]:
        """
        Return posterior distribution of S given:
        - my revealed sum (obs.k_mine)
        - foresight sum and count (if any)
        """
        # Number of coins we have not seen: total 40 - (my revealed count) - (foresight count)
        my_revealed_count = len(obs.my_revealed)
        unknown_count = 40 - my_revealed_count - self.foresight_count
        known_sum = obs.k_mine + self.foresight_sum
        dist = self._get_dist(unknown_count)
        # Shift by known_sum
        posterior = {}
        for s_unknown, prob in dist.items():
            s = known_sum + s_unknown
            posterior[s] = prob
        return posterior

    def _posterior_mean(self, obs: Any) -> float:
        """Expected value of S given current information."""
        dist = self._posterior_distribution(obs)
        mean = sum(s * p for s, p in dist.items())
        return mean

    def _prob_inside(self, obs: Any, bid: int, ask: int) -> float:
        """Probability that S lies within [bid, ask] under posterior."""
        dist = self._posterior_distribution(obs)
        prob = 0.0
        for s, p in dist.items():
            if bid <= s <= ask:
                prob += p
        return prob

    def _estimate_power_value(self, power: str, obs: Any) -> float:
        """
        Estimate the expected PnL benefit (in ticks) of winning this power now.
        Uses analytic approximations.
        """
        if power == "FORESIGHT":
            # Value of seeing up to 16 opponent coins.
            # Each revealed coin reduces variance of S by 1.
            # The benefit is roughly the reduction in expected absolute error,
            # which scales with sqrt(variance). We approximate.
            n_see = min(16, 4 * obs.round)
            # Variance of S now: unknown_count (since each coin has variance 1)
            unknown = 40 - 4 * obs.round  # before foresight
            var_before = unknown
            var_after = unknown - n_see
            # Improvement in precision: 1/sqrt(var_after) - 1/sqrt(var_before)
            # Convert to ticks: each tick is ~1 unit of S, so benefit ~ sqrt reduction.
            if var_before > 0 and var_after > 0:
                improvement = (1 / math.sqrt(var_after) - 1 / math.sqrt(var_before)) * 10
            else:
                improvement = 0
            return max(0, min(3.0, improvement))
        elif power == "TRICK_ROOM":
            # Magnitude 3, applied only if forced fill.
            # Estimate probability of forced fill based on typical negotiation.
            # We'll approximate 0.35.
            prob_forced = 0.35
            return 3.0 * prob_forced
        elif power == "STEALTH_ROCK":
            # Magnitude 2, persistent for remaining rounds.
            # Probability of forced fill per round ~ 0.35.
            remaining_rounds = 5 - obs.round + 1  # includes current round
            if obs.round == 5:
                return 0.0  # not eligible anyway
            return 2.0 * 0.35 * remaining_rounds
        elif power == "SUBSTITUTE":
            # Loss cap of 2 ticks.
            # Value is like a put option: expected payoff = E[max(0, -profit - 2)]?
            # We can approximate as 0.5 ticks.
            return 0.5
        elif power == "TRANSFORM":
            # Option to swap hands. Value depends on our hand quality.
            # If our revealed sum is negative, swap may be beneficial.
            # Approximate value as 0.2 * abs(obs.k_mine) / 20? But capped.
            if obs.k_mine < 0:
                return 0.3 * (-obs.k_mine) / 20
            else:
                return 0.0
        return 0.0