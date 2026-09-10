# Name: Heet Nisar
# College: Dwarkadas J. Sanghvi College of Engineering
# Roll Number: 60005240044

import random


POWER_VALUES = {
    "FORESIGHT": {
        1: 0.76,
        2: 1.16,
        3: 1.48,
        4: 1.97,
        5: 2.02,
    },
    "TRICK_ROOM": {
        1: 1.14,
        2: 0.00,
        3: 0.00,
        4: 0.60,
        5: 0.52,
    },
    "SUBSTITUTE": {
        1: 1.46,
        2: 1.15,
        3: 0.95,
        4: 0.57,
        5: 0.29,
    },
    "STEALTH_ROCK": {
        1: 1.51,
        2: 0.75,
        3: 0.75,
        4: 0.75,
        5: 0.00,
    },
    "TRANSFORM": {
        1: 1.58,
        2: 1.24,
        3: 1.31,
        4: 0.00,
        5: 0.00,
    },
}

SHADE = 0.60
MIN_SHADE = 0.50
MAX_SHADE = 0.68

FLAT_THRESHOLD = 1
OPP_FLAT_THRESHOLD = 2.0

FORCE_FEE = 2.0
FORCE_MARGIN = 0.20

SHIFT_POWER = {
    "TRICK_ROOM": 3,
    "STEALTH_ROCK": 2,
}


class Bot:
    name = "HeetOracle"

    def reset(self, seat, config, seed):
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)

        self.opp_openings = {}
        self.opp_bias = 0.0
        self.opp_bias_n = 0

        self.best_foresight_n = 0
        self.best_foresight_sum = 0

    def _update_foresight(self, obs):
        if not obs.foresight:
            return

        n = len(obs.foresight)

        if n >= self.best_foresight_n:
            self.best_foresight_n = n
            self.best_foresight_sum = sum(obs.foresight)

    def _opponent_anchor(self, obs):
        earlier = [
            r for r in self.opp_openings
            if r < obs.round
        ]

        if not earlier:
            return None

        r = max(earlier)
        return self.opp_openings[r]

    def _record_opening(self, obs, quote):
        if obs.is_maker:
            return

        if obs.round in self.opp_openings:
            return

        midpoint = (quote[0] + quote[1]) / 2.0

        self.opp_openings[obs.round] = midpoint

        foresight_sum = None

        if obs.foresight:
            foresight_sum = sum(obs.foresight)

        if foresight_sum is not None:
            error = midpoint - foresight_sum

            self.opp_bias_n += 1
            n = self.opp_bias_n

            self.opp_bias += (error - self.opp_bias) / n

    def _estimate_opponent_k(self, obs, quote=None):
        self._update_foresight(obs)

        signals = []

        if self.best_foresight_n > 0:
            remaining = max(
                1,
                20 - self.best_foresight_n
            )

            signals.append(
                (
                    float(self.best_foresight_sum),
                    float(remaining)
                )
            )

        anchor = self._opponent_anchor(obs)

        if anchor is not None:
            corrected = anchor - self.opp_bias

            remaining = max(
                1,
                20 - 4 * max(self.opp_openings)
            )

            signals.append(
                (
                    float(corrected),
                    float(remaining)
                )
            )

        if quote is not None and not obs.is_maker:
            midpoint = (quote[0] + quote[1]) / 2.0

            corrected = midpoint - self.opp_bias

            remaining = max(
                1,
                20 - 4 * obs.round
            )

            signals.append(
                (
                    float(corrected),
                    float(remaining)
                )
            )

        if not signals:
            return 0.0

        weight_sum = 0.0
        value_sum = 0.0

        for value, variance in signals:
            weight = 1.0 / variance
            value_sum += value * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.0

        return value_sum / weight_sum

    def _estimate_score(self, obs, quote=None):
        opponent_k = self._estimate_opponent_k(obs, quote)

        return float(obs.k_mine) + opponent_k

    def _power_value(self, obs, power):
        return POWER_VALUES.get(
            power,
            {}
        ).get(
            obs.round,
            0.50
        )

    def _transform_value(self, obs):
        value = self._power_value(
            obs,
            "TRANSFORM"
        )

        if value <= 0:
            return 0.0

        if abs(obs.k_mine) <= FLAT_THRESHOLD:
            return value

        opponent_k = self._opponent_anchor(obs)

        if opponent_k is not None:
            if abs(opponent_k) <= OPP_FLAT_THRESHOLD:
                return value * 0.20

        return 0.0

    def _future_te_value(self, obs):
        remaining_rounds = 5 - obs.round

        if remaining_rounds <= 0:
            return 0.0

        future = []

        for power in POWER_VALUES:
            for r in range(
                obs.round + 1,
                6
            ):
                if r in POWER_VALUES[power]:
                    future.append(
                        POWER_VALUES[power][r]
                    )

        if not future:
            return 0.0

        future.sort(reverse=True)

        return future[0]

    def _shade(self, obs, value):
        shade = SHADE

        if obs.te_mine <= 5:
            shade = MIN_SHADE

        elif obs.te_mine >= 18:
            shade = MAX_SHADE

        if obs.round >= 4:
            shade += 0.03

        if value >= 1.5:
            shade += 0.02

        return max(
            MIN_SHADE,
            min(MAX_SHADE, shade)
        )

    def bid(self, obs, offered):
        if not offered:
            return {}

        if obs.te_mine <= 0:
            return {}

        power = offered[0]

        if power == "TRANSFORM":
            value = self._transform_value(obs)
        else:
            value = self._power_value(
                obs,
                power
            )

        if value <= 0:
            return {}

        future_value = self._future_te_value(obs)

        if obs.round < 5:
            value -= 0.05 * future_value

        value = max(0.0, value)

        fair_te = value / self.config.TE_SALVAGE

        shade = self._shade(
            obs,
            value
        )

        bid = int(
            fair_te * shade
        )

        if power == "FORESIGHT":
            if obs.round >= 4:
                bid += 1

        if power == "SUBSTITUTE":
            if obs.round <= 2:
                bid += 1

        if power == "TRANSFORM":
            if abs(obs.k_mine) <= 1:
                bid += 1

        bid = max(
            0,
            min(
                bid,
                obs.te_mine
            )
        )

        if bid <= 0:
            return {}

        return {
            power: bid
        }

    def quote(self, obs):
        value = round(
            self._estimate_score(obs)
        )

        width = obs.final_cap

        lo = value - width // 2
        hi = lo + width

        return (
            lo,
            hi
        )

    def _shift_for_me(self, obs):
        mine = 0
        theirs = 0

        if "TRICK_ROOM" in obs.powers_mine:
            mine += SHIFT_POWER["TRICK_ROOM"]

        if "STEALTH_ROCK" in obs.powers_mine:
            mine += SHIFT_POWER["STEALTH_ROCK"]

        if "TRICK_ROOM" in obs.powers_theirs:
            theirs += SHIFT_POWER["TRICK_ROOM"]

        if "STEALTH_ROCK" in obs.powers_theirs:
            theirs += SHIFT_POWER["STEALTH_ROCK"]

        return mine - theirs

    def _buy_ev(self, value, ask):
        return value - ask

    def _sell_ev(self, value, bid):
        return bid - value

    def _counter_range(self, obs, bid, ask, value):
        width = ask - bid

        max_width = min(
            width,
            max(
                obs.final_cap,
                width - self.config.MIN_REDUCTION
            )
        )

        if max_width <= 0:
            return (
                bid,
                ask
            )

        center = round(value)

        low = center - max_width // 2
        high = low + max_width

        if low < bid:
            low = bid
            high = low + max_width

        if high > ask:
            high = ask
            low = high - max_width

        low = max(
            bid,
            low
        )

        high = min(
            ask,
            high
        )

        if high - low < obs.final_cap:
            high = ask
            low = ask - obs.final_cap

            if low < bid:
                low = bid
                high = bid + obs.final_cap

        return (
            int(low),
            int(high)
        )

    def _final_force_ev(
        self,
        obs,
        bid,
        ask,
        value
    ):
        width = ask - bid

        max_width = min(
            width,
            max(
                obs.final_cap,
                width - self.config.MIN_REDUCTION
            )
        )

        low = ask - max_width
        high = ask

        midpoint = (
            low + high
        ) // 2

        shift = self._shift_for_me(obs)

        forced_price = midpoint + shift

        ev = forced_price - value

        ev -= FORCE_FEE

        if "SUBSTITUTE" in obs.powers_mine:
            ev = max(
                ev,
                -2.0
            )

        return ev, low, high

    def respond(self, obs, quote, turn):
        self._record_opening(
            obs,
            quote
        )

        bid = quote[0]
        ask = quote[1]

        value = self._estimate_score(
            obs,
            quote
        )

        buy_ev = self._buy_ev(
            value,
            ask
        )

        sell_ev = self._sell_ev(
            value,
            bid
        )

        if "SUBSTITUTE" in obs.powers_mine:
            buy_threshold = -0.75
            sell_threshold = -0.75
        else:
            buy_threshold = 0.0
            sell_threshold = 0.0

        if turn == obs.n_turns:
            force_ev, force_bid, force_ask = (
                self._final_force_ev(
                    obs,
                    bid,
                    ask,
                    value
                )
            )

            best_accept = max(
                buy_ev if buy_ev >= buy_threshold else -999.0,
                sell_ev if sell_ev >= sell_threshold else -999.0
            )

            if force_ev > best_accept + FORCE_MARGIN:
                return (
                    "COUNTER",
                    force_bid,
                    force_ask
                )

            if buy_ev >= sell_ev:
                return "ACCEPT_BUY"

            return "ACCEPT_SELL"

        if buy_ev >= buy_threshold:
            if buy_ev >= sell_ev:
                return "ACCEPT_BUY"

        if sell_ev >= sell_threshold:
            if sell_ev > buy_ev:
                return "ACCEPT_SELL"

        new_bid, new_ask = self._counter_range(
            obs,
            bid,
            ask,
            value
        )

        if new_bid >= new_ask:
            if buy_ev >= sell_ev:
                return "ACCEPT_BUY"

            return "ACCEPT_SELL"

        return (
            "COUNTER",
            new_bid,
            new_ask
        )

    def use_transform(self, obs):
        if abs(obs.k_mine) <= FLAT_THRESHOLD:
            return True

        opponent_k = self._opponent_anchor(
            obs
        )

        if opponent_k is not None:
            if abs(opponent_k) > OPP_FLAT_THRESHOLD:
                return False

        return False