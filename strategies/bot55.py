import random

# REDUCED R5 SHADE: Prevents overpaying for powers against elite opponents
SHADE_BY_ROUND = {1: 0.60, 2: 0.70, 3: 0.80, 4: 0.90, 5: 0.75}

POWER_VALUES = {
    "FORESIGHT":    {1: 0.76, 2: 1.16, 3: 1.48, 4: 1.97, 5: 2.02},
    "TRICK_ROOM":   {1: 1.14, 2: 0.00, 3: 0.00, 4: 0.60, 5: 0.52},
    "SUBSTITUTE":   {1: 1.46, 2: 1.15, 3: 0.95, 4: 0.57, 5: 0.29},
    "STEALTH_ROCK": {1: 1.51, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.00},
    "TRANSFORM":    {1: 1.58, 2: 1.24, 3: 1.31, 4: 0.00, 5: 0.00},
}

TRANSFORM_THRESHOLD = 0.5
Z_FORCE = 0.55
FORCE_MARGIN_FLOOR = 0.3
SHIFT_POWERS = {"TRICK_ROOM": 3, "STEALTH_ROCK": 2}


class Bot:
    name = "MaxOracle"

    def reset(self, seat, config, seed) -> None:
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)
        self._opp_anchor = {}
        self._foresight_n = 0
        self._foresight_sum = 0

    # =============================================
    # VALUE ESTIMATION (UPGRADED FOR R5)
    # =============================================

    def _update_foresight(self, obs):
        if obs.foresight:
            n = len(obs.foresight)
            if n >= self._foresight_n:
                self._foresight_n = n
                self._foresight_sum = sum(obs.foresight)

    def _best_anchor(self, obs):
        earlier = [r for r in self._opp_anchor if r < obs.round]
        if not earlier:
            return None, None
        r = max(earlier)
        return self._opp_anchor[r], r

    def _estimate_S(self, obs, live_quote=None):
        n_private = self.config.N_PRIVATE
        per_round = self.config.REVEAL_PER_ROUND
        self._update_foresight(obs)

        components = [(0.0, float(n_private))]  # (mean, variance)
        if self._foresight_n > 0:
            var = max(1.0, n_private - self._foresight_n)
            components.append((self._foresight_sum, var))

        anchor, anchor_r = self._best_anchor(obs)
        if anchor is not None:
            anchor_var = max(1.0, n_private - per_round * anchor_r)
            # --- CRITICAL FIX 1: De-weight opponent's anchor in Round 5 ---
            # Top models fake R4 quotes to trick us into overpaying in R5.
            if obs.round == 5:
                anchor_var *= 5.0  # Increases variance, lowers trust in R4 anchor
            components.append((anchor, anchor_var))

        if live_quote is not None and not obs.is_maker:
            live_mid = (live_quote[0] + live_quote[1]) / 2.0
            var = max(1.0, n_private - per_round * obs.round)
            components.append((live_mid, var))

        w_sum = sum(1.0 / v for _, v in components)
        k_theirs_hat = sum(m / v for m, v in components) / w_sum
        opp_var = 1.0 / w_sum

        my_unseen_var = max(0, n_private - len(obs.my_revealed))
        S_var = opp_var + my_unseen_var
        return obs.k_mine + k_theirs_hat, S_var

    # =============================================
    # AUCTION BIDDING
    # =============================================

    def bid(self, obs, offered: list) -> dict:
        if not offered or obs.te_mine <= 0:
            return {}

        values = {}
        for name in offered:
            shade = SHADE_BY_ROUND.get(obs.round, 0.60)
            if name == "TRANSFORM":
                v = self._transform_value(obs)
            else:
                v = POWER_VALUES.get(name, {}).get(obs.round, 0.5)
            if v > 0:
                values[name] = v

        if not values:
            return {}

        fair_te = {n: v / self.config.TE_SALVAGE for n, v in values.items()}
        raw_bid = {n: int(f * SHADE_BY_ROUND.get(obs.round, 0.60)) for n, f in fair_te.items()}

        total = sum(raw_bid.values())
        if total <= obs.te_mine:
            return {n: b for n, b in raw_bid.items() if b > 0}

        budget = obs.te_mine
        out = {}
        for n in sorted(values, key=lambda k: values[k], reverse=True):
            amt = min(raw_bid[n], budget)
            if amt > 0:
                out[n] = amt
                budget -= amt
        return out

    def _transform_value(self, obs):
        swap = POWER_VALUES["TRANSFORM"].get(obs.round, 0.0)
        if abs(obs.k_mine) <= TRANSFORM_THRESHOLD:
            return swap
        return 0.0

    # =============================================
    # NEGOTIATION
    # =============================================

    def quote(self, obs) -> tuple:
        S_hat, _ = self._estimate_S(obs)
        v = round(S_hat)
        cap = obs.final_cap
        lo = v - cap // 2
        return (lo, lo + cap)

    def respond(self, obs, quote: tuple, turn: int):
        if turn == 2:
            self._opp_anchor[obs.round] = (quote[0] + quote[1]) / 2.0

        bid_p, ask_p = quote
        live_quote = quote if turn == 2 else None
        S_hat, S_var = self._estimate_S(obs, live_quote)
        se = S_var ** 0.5

        substitute = "SUBSTITUTE" in obs.powers_mine
        loss_cap = getattr(self.config, 'LOSS_CAP', -1.5)
        thresh = loss_cap if substitute else 0.0

        edge_buy = S_hat - ask_p
        edge_sell = bid_p - S_hat

        if turn == obs.n_turns:
            return self._final_turn(obs, bid_p, ask_p, S_hat, se, edge_buy, edge_sell,
                                     substitute, loss_cap)

        if edge_buy > thresh and edge_buy >= edge_sell:
            return "ACCEPT_BUY"
        if edge_sell > thresh:
            return "ACCEPT_SELL"

        width = ask_p - bid_p
        max_width = min(width, max(obs.final_cap, width - self.config.MIN_REDUCTION))
        center = max(bid_p, min(round(S_hat), ask_p - max_width))
        return ("COUNTER", center, center + max_width)

    def _final_turn(self, obs, bid_p, ask_p, S_hat, se, edge_buy, edge_sell, substitute, loss_cap):
        width = ask_p - bid_p
        max_width = min(width, max(obs.final_cap, width - self.config.MIN_REDUCTION))

        # --- CRITICAL FIX 2: Center your Final Turn Counter on YOUR Value ---
        # Previous bug anchored it to their Ask, making you overpay on forced fills.
        center = round(S_hat)
        lo = int(max(bid_p, center - max_width // 2))
        hi = int(min(ask_p, center + max_width // 2))

        shift = 0
        for power, mag in SHIFT_POWERS.items():
            if power in obs.powers_mine:
                shift += mag
            if power in obs.powers_theirs:
                shift -= mag

        forced_price = (lo + hi) // 2 + shift
        ev_force = (forced_price - S_hat) - self.config.FORCED_FILL_FEE

        ev_buy, ev_sell = edge_buy, edge_sell
        if substitute:
            ev_buy = max(ev_buy, loss_cap)
            ev_sell = max(ev_sell, loss_cap)
            ev_force = max(ev_force, loss_cap)

        best_accept = max(ev_buy, ev_sell)
        force_margin = max(FORCE_MARGIN_FLOOR, Z_FORCE * se)
        if ev_force > best_accept + force_margin:
            return ("COUNTER", lo, hi)

        if ev_buy >= ev_sell:
            return "ACCEPT_BUY"
        return "ACCEPT_SELL"

    def use_transform(self, obs) -> bool:
        return abs(obs.k_mine) <= TRANSFORM_THRESHOLD