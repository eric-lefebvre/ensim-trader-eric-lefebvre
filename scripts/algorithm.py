"""
Décision automatique pour l'ENSIM Trade Center (même algorithme que le dashboard docs/index.html).

1. Modèle : pour chaque produit, prix(t) = (niveau + dérive·t) × (1 + k·sin(2πt/P + φ)), plancher 0,01
   (forme utilisée par scripts/market_tick.py de l'organisateur ; paramètres estimés sur l'historique).
2. Règle : trois lignes qui suivent la tendance (moyenne, ligne des creux, ligne des sommets).
   Achat de 100 u si le cours passe sous « ligne des creux + s_achat × chemin vers la moyenne »,
   vente de 100 u s'il passe au-dessus de « ligne des sommets − s_vente × chemin vers la moyenne ».
   Un seul argent partagé, ordre du marché A puis B puis C, achats limités par l'argent restant
   moins la réserve de 50 (MIN_MONEY_RESERVE). Le signal passe ensuite dans apply_budget_guard (manuel).
3. Seuils : recherchés par blocs de L ticks (multiple commun des périodes, ex. 96), sur N_BLOCKS blocs,
   pour maximiser la valeur finale prévue. La décision du prochain tick utilise les seuils du bloc 1.

Écrit decision.json (lu par le marché) et docs/data/strategy.json (lu par le dashboard).
Ne fait rien si le marché n'a pas avancé depuis le dernier calcul (sauf FORCE=1).
"""
import json, math, os, time
import numpy as np
import requests

ORGANIZER = "exofri/ensim-trade-center"
TRADER = "eric-lefebvre"
MARKET_PATH = "docs/data/market.json"
HOLDINGS_PATH = f"traders/{TRADER}/holdings.json"
MARKET_URL = f"https://raw.githubusercontent.com/{ORGANIZER}/refs/heads/main/{MARKET_PATH}"
HOLDINGS_URL = f"https://raw.githubusercontent.com/{ORGANIZER}/refs/heads/main/{HOLDINGS_PATH}"
STRATEGY_FILE = "docs/data/strategy.json"
N_BLOCKS = 3          # horizon de recherche = 3 tranches (288 ticks avec des tranches de 96)
LOT = 100             # unités max par tick et par produit (règle du marché)
GRID = np.round(np.arange(21) * 0.05, 2)          # seuils 0 %, 5 %, ..., 100 %
GRID_B = np.repeat(GRID, 21)                       # 441 paires (achat, vente), même ordre que le dashboard
GRID_S = np.tile(GRID, 21)


# ---------------------------------------------------------------- données
MIN_MONEY_RESERVE = 50 # never let a buy signal spend below this much money

# Produits dont le seuil de vente doit rester inférieur ou égal au seuil d'achat : leur zone de vente n'est
# jamais plus large que leur zone d'achat, donc sur un cycle on achète au moins autant qu'on vend (stocker du B).
SELL_LE_BUY = {"B"}

def allowed(p):
    """Paires (achat, vente) autorisées pour p parmi les 441 de la grille (None = toutes)."""
    return (GRID_S <= GRID_B + 1e-9) if p in SELL_LE_BUY else None

# --- fourni par le manuel (TP_2_STUDENT_MANUAL), inchangé ---
def fetch_json(url):
    resp = requests.get(url, proxies={"http": None, "https": None}, timeout=(10, 20))
    resp.raise_for_status()
    return resp.json()

# --- fourni par le manuel (TP_2_STUDENT_MANUAL), inchangé ---
def apply_budget_guard(raw_signal, money, price, reserve=MIN_MONEY_RESERVE):
    # A buy signal (positive) that would spend into the reserve gets scaled down,
    # not silently zeroed out -- you still act on the signal, just smaller. A sell
    # signal is never touched here: selling only ever adds money, never risks it.
    if raw_signal <= 0:
        return raw_signal
    if money <= reserve:
        return 0.0
    max_affordable_units = (money - reserve) / price
    desired_units = 100 * raw_signal
    if desired_units <= max_affordable_units:
        return raw_signal
    return round(max_affordable_units / 100, 3)

def fetch_api_json(url, headers):
    resp = requests.get(url, headers=headers, proxies={"http": None, "https": None}, timeout=(10, 20))
    resp.raise_for_status()
    return resp.json()

def fetch_feeds():
    """Lit market.json et holdings.json au même commit (le dernier qui a modifié market.json),
    pour éviter le cache de raw.githubusercontent.com (jusqu'à 5 min de retard)."""
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        commits = fetch_api_json(f"https://api.github.com/repos/{ORGANIZER}/commits?path={MARKET_PATH}&per_page=1", headers)
        sha = commits[0]["sha"]
        base = f"https://raw.githubusercontent.com/{ORGANIZER}/{sha}"
        return fetch_json(f"{base}/{MARKET_PATH}"), fetch_json(f"{base}/{HOLDINGS_PATH}"), sha
    except Exception as e:
        print(f"API GitHub indisponible ({e}) -- lecture directe des URL raw")
        return fetch_json(MARKET_URL), fetch_json(HOLDINGS_URL), None


# ---------------------------------------------------------------- modèle
class Model:
    def __init__(self, y):
        y = np.asarray(y, dtype=float)
        n = len(y)
        t = np.arange(n, dtype=float)
        # 1) période : balayage de P (pas 0,25), modèle a + b t + c sin + d cos
        best = None
        max_p = min(300, n * 0.9)
        k = 0
        while 6 + 0.25 * k <= max_p:
            P = 6 + 0.25 * k
            w = 2 * math.pi / P
            X = np.column_stack([np.ones(n), t, np.sin(w * t), np.cos(w * t)])
            c = np.linalg.solve(X.T @ X, X.T @ y)
            sse = float(((X @ c - y) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, P, w, c)
            k += 1
        _, P, w, c = best
        a, b = c[0], c[1]
        si, co = np.sin(w * t), np.cos(w * t)
        ks = kc = 0.0
        # 2) moindres carrés alternés : oscillation relative, puis niveau et dérive
        for _ in range(30):
            r = y / (a + b * t) - 1
            ks, kc = np.linalg.solve([[si @ si, si @ co], [si @ co, co @ co]], [r @ si, r @ co])
            g = 1 + ks * si + kc * co
            a, b = np.linalg.solve([[g @ g, (t * g) @ g], [(t * g) @ g, (t * t * g) @ g]], [y @ g, (y * t) @ g])
        self.a, self.b, self.P, self.w = float(a), float(b), float(P), float(w)
        self.k, self.phi = float(math.hypot(ks, kc)), float(math.atan2(kc, ks))

    def trend(self, t):
        return np.maximum(self.a + self.b * np.asarray(t, dtype=float), 0)

    def f(self, t):
        t = np.asarray(t, dtype=float)
        return np.maximum((self.a + self.b * t) * (1 + self.k * np.sin(self.w * t + self.phi)), 0.01)


BLOCK_TICKS = 96   # longueur fixe d'une tranche (1 cycle de B, 2 de A, 4 de C) : jamais plus de 96 ticks

def block_length(models):
    """Tranche fixée à 96 ticks. On signale seulement si les périodes estimées ne s'y emboîtent plus."""
    for p, m in models.items():
        n = max(1, round(BLOCK_TICKS / m.P))
        if abs(BLOCK_TICKS - n * m.P) > 0.03 * BLOCK_TICKS:
            print(f"Attention : période de {p} estimée à {m.P:.2f}, ne tient pas un nombre entier de fois dans {BLOCK_TICKS} ticks")
    return BLOCK_TICKS


# ---------------------------------------------------------------- simulation vectorisée
def precompute(models, ps, t0, T):
    i = np.arange(T + 1, dtype=float)
    pre = {}
    for p in ps:
        m = models[p]
        tr = m.trend(t0 + i)
        pre[p] = {"pr": m.f(t0 + i), "tr": tr, "low": tr * (1 - m.k), "high": tr * (1 + m.k)}
    return pre

def simulate(ps, plan, L, pre, i0, money, units, T, vary=None):
    """Simule les ticks i0+1 .. T. plan = liste de {p: (b, s)} par bloc de L ticks.
    vary = (bloc, produit) : ce couple prend les 441 paires de GRID_B / GRID_S en parallèle
    (bloc None = tous les blocs). Renvoie (valeur finale, argent, stocks) en tableaux."""
    size = len(GRID_B) if vary else 1
    money = np.full(size, money, dtype=float) if np.isscalar(money) else money.copy()
    units = {p: (np.full(size, units[p], dtype=float) if np.isscalar(units[p]) else units[p].copy()) for p in ps}
    val = np.zeros(size)
    for i in range(i0 + 1, T + 1):
        blk = min(len(plan) - 1, (i - 1) // L)
        val = np.zeros(size)
        for p in ps:
            d = pre[p]
            pr, tr, low, high = d["pr"][i], d["tr"][i], d["low"][i], d["high"][i]
            if vary and vary[1] == p and (vary[0] is None or vary[0] == blk):
                b, s = GRID_B, GRID_S
            else:
                b, s = plan[blk][p]
            if tr > 0:
                buy = pr <= low + b * (tr - low)
                sell = ~buy & (pr >= high - s * (high - tr))
                qb = np.where(buy, np.minimum(LOT, np.maximum(0.0, (money - MIN_MONEY_RESERVE) / pr)), 0.0)
                qs = np.where(sell, np.minimum(LOT, units[p]), 0.0)
                money = money - qb * pr + qs * pr
                units[p] = units[p] + qb - qs
            val = val + units[p] * pr
    return money + val, money, units


def optimize_fixed(ps, pre, money0, units0, T):
    """Mêmes seuils sur toute la durée : chaque produit seul, puis tour à tour avec l'argent partagé."""
    big = T + 1
    th = {}
    for p in ps:
        v, _, _ = simulate([p], [{p: (0.2, 0.2)}], big, pre, 0, money0, {p: units0[p]}, T, vary=(None, p))
        if allowed(p) is not None:
            v = np.where(allowed(p), v, -np.inf)
        j = int(np.argmax(v))
        th[p] = (float(GRID_B[j]), float(GRID_S[j]))
    best = simulate(ps, [th], big, pre, 0, money0, units0, T)[0][0]
    for _ in range(4):
        improved = False
        for p in ps:
            v, _, _ = simulate(ps, [th], big, pre, 0, money0, units0, T, vary=(None, p))
            if allowed(p) is not None:
                v = np.where(allowed(p), v, -np.inf)
            j = int(np.argmax(v))
            if v[j] > best + 1e-6:
                best, th[p], improved = float(v[j]), (float(GRID_B[j]), float(GRID_S[j])), True
        if not improved:
            break
    return th, float(best)

MIN_GAIN = 0.0005   # on ne change les seuils d'une tranche que si la valeur finale gagne plus de 0,05 %

def descend(ps, plan, pre, money0, units0, L, dying):
    """Amélioration tranche par tranche, produit par produit (max 3 tours), à partir du plan donné.
    - Stabilité : un changement n'est retenu que s'il améliore la valeur finale de plus de MIN_GAIN.
    - Produit dont la tendance atteint 0 pendant l'horizon (ex. C) : d'une tranche à la suivante, le seuil
      d'achat ne peut que baisser et le seuil de vente que monter (on pousse de plus en plus à la vente)."""
    nb = len(plan)
    T = L * nb
    plan = [dict(b) for b in plan]
    best = simulate(ps, plan, L, pre, 0, money0, units0, T)[0][0]
    for _ in range(3):
        improved = False
        for k in range(nb):
            _, m_k, u_k = simulate(ps, plan, L, pre, 0, money0, units0, k * L)   # état au début de la tranche k
            m_k, u_k = float(m_k[0]), {p: float(u_k[p][0]) for p in ps}
            for p in ps:
                v, _, _ = simulate(ps, plan, L, pre, k * L, m_k, u_k, T, vary=(k, p))
                if p in dying:
                    ok = np.ones(len(v), dtype=bool)
                    if k > 0:
                        ok &= (GRID_B <= plan[k - 1][p][0] + 1e-9) & (GRID_S >= plan[k - 1][p][1] - 1e-9)
                    if k < nb - 1:
                        ok &= (GRID_B >= plan[k + 1][p][0] - 1e-9) & (GRID_S <= plan[k + 1][p][1] + 1e-9)
                    v = np.where(ok, v, -np.inf)
                if allowed(p) is not None:
                    v = np.where(allowed(p), v, -np.inf)
                j = int(np.argmax(v))
                if v[j] > best * (1 + MIN_GAIN):
                    best, plan[k][p], improved = float(v[j]), (float(GRID_B[j]), float(GRID_S[j])), True
        if not improved:
            break
    return plan, float(best)

def optimize_blocks(ps, pre, money0, units0, L, nb):
    """Seuils propres à chaque tranche de L ticks. Pour ne pas rester bloqué sur une solution « assez bonne »,
    la recherche part de plusieurs plans et garde le meilleur :
      1) les meilleurs seuils identiques sur tout l'horizon ;
      2) tranche 1 = meilleurs seuils sur 1 seule tranche, tranches suivantes = 1) ;
      3) les meilleurs seuils sur 1 seule tranche, recopiés dans toutes les tranches.
    (pour un produit qui tend vers 0, les départs gardent des seuils identiques dans toutes les tranches)."""
    T = L * nb
    fixed, fixed_value = optimize_fixed(ps, pre, money0, units0, T)
    dying = {p for p in ps if (pre[p]["tr"][1:] <= 0).any()}
    starts = [[dict(fixed) for _ in range(nb)]]
    if nb > 1:
        one, _, _ = optimize_blocks(ps, pre, money0, units0, L, 1)
        first = {p: (fixed[p] if p in dying else one[0][p]) for p in ps}
        starts.append([first] + [dict(fixed) for _ in range(nb - 1)])
        starts.append([dict(first) for _ in range(nb)])
    best_plan, best = None, -np.inf
    for start in starts:
        plan, value = descend(ps, start, pre, money0, units0, L, dying)
        if value > best + 1e-6:
            best_plan, best = plan, value
    return best_plan, float(best), fixed_value


def next_decision(models, ps, plan, last, money, units):
    """Décision du prochain tick (identique au 1er tick de la simulation et au dashboard)."""
    dec, reasons = {}, {}
    units = dict(units)
    for p in ps:
        m = models[p]
        t = last + 1
        pr, tr = float(m.f(t)), float(m.trend(t))
        low, high = tr * (1 - m.k), tr * (1 + m.k)
        b, s = plan[0][p]
        buy_at, sell_at = low + b * (tr - low), high - s * (high - tr)
        q = 0.0
        if tr > 0 and pr <= buy_at:
            q = min(LOT, max(0.0, (money - MIN_MONEY_RESERVE) / pr))
            money -= q * pr
            units[p] += q
            reasons[p] = f"cours prévu {pr:.3f} <= seuil d'achat {buy_at:.3f}"
        elif tr > 0 and pr >= sell_at:
            q = -min(LOT, units[p])
            money -= q * pr
            units[p] += q
            reasons[p] = f"cours prévu {pr:.3f} >= seuil de vente {sell_at:.3f}"
        else:
            reasons[p] = f"cours prévu {pr:.3f} entre {buy_at:.3f} et {sell_at:.3f}"
        dec[p] = math.floor(q + 0.5) / 100     # comme Math.round(q) / 100 dans le dashboard
    return dec, reasons


# ---------------------------------------------------------------- signal (contrat du manuel)
_strategy_cache = {}

def build_strategy(history, holdings, products):
    """Modèle + seuils par tranche + signal du prochain tick, calculés une fois pour les 3 produits
    (l'argent est partagé entre eux, donc les signaux se décident ensemble)."""
    ps = [p for p in products if len(history) >= 12]
    models = {p: Model([row[p] for row in history]) for p in ps}
    last = len(history) - 1
    L = block_length(models)
    nb = N_BLOCKS
    money0 = float(holdings["money"])
    units0 = {p: float(holdings["products"][p]) for p in ps}
    pre = precompute(models, ps, last, L * nb)
    plan, value, fixed_value = optimize_blocks(ps, pre, money0, units0, L, nb)
    dec, reasons = next_decision(models, ps, plan, last, money0, units0)
    return {"models": models, "L": L, "nb": nb, "plan": plan, "value": value,
            "fixed_value": fixed_value, "signals": dec, "reasons": reasons}

def compute_signal(prices, product, history, holdings, products):
    """Signal brut dans [-1, 1] pour un produit (signature étendue comme le permet le manuel :
    historique des 3 produits et holdings, car l'argent est partagé). Passé ensuite dans
    apply_budget_guard par main(), comme dans le squelette du manuel."""
    if len(prices) < 12:
        return 0.0
    if "s" not in _strategy_cache:
        _strategy_cache["s"] = build_strategy(history, holdings, products)
    return max(-1.0, min(1.0, _strategy_cache["s"]["signals"].get(product, 0.0)))


# ---------------------------------------------------------------- main
def main():
    market, holdings, sha = fetch_feeds()
    history = market["history"]
    tick = history[-1]["t"]

    previous = {}
    if os.path.exists(STRATEGY_FILE):
        with open(STRATEGY_FILE) as fh:
            previous = json.load(fh)
    if previous.get("tick") == tick and previous.get("market_generated_at") == market["generated_at"] and os.environ.get("FORCE") != "1":
        print(f"Pas de nouveau tick (dernier : {tick}) -- rien à faire.")
        return

    started = time.time()
    products = market["products"]
    decision = {}
    for p in products:   # boucle du squelette du manuel : signal puis garde budgétaire
        prices = [row[p] for row in history]
        raw = compute_signal(prices, p, history, holdings, products)
        current_price = prices[-1] if prices else 0
        decision[p] = round(apply_budget_guard(raw, holdings["money"], current_price), 3)
        if not math.isfinite(decision[p]):   # filet de sécurité : jamais de NaN / infini dans decision.json
            print(f"Attention : signal non fini pour {p} -> 0")
            decision[p] = 0.0
    decision["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open("decision.json", "w") as fh:
        json.dump(decision, fh, indent=2)

    st = _strategy_cache.get("s")
    if st is None:
        print(f"Historique trop court -- decision.json : {decision}")
        return
    models, L, nb, plan, ps = st["models"], st["L"], st["nb"], st["plan"], list(st["models"])
    value, fixed_value, dec, reasons = st["value"], st["fixed_value"], st["signals"], st["reasons"]

    strategy = {
        "generated_at": decision["updated_at"],
        "tick": tick,
        "market_generated_at": market["generated_at"],
        "market_commit": sha,
        "block_ticks": L,
        "blocks": nb,
        "plan": [{p: {"buy": b[p][0], "sell": b[p][1]} for p in ps} for b in plan],
        "signal_raw": dec,
        "decision": {p: decision[p] for p in products},
        "reasons": reasons,
        "expected_value": round(value, 2),
        "expected_value_fixed_thresholds": round(fixed_value, 2),
        "models": {p: {"period": m.P, "level": m.a, "drift": m.b, "amplitude": m.k, "phase": m.phi} for p, m in models.items()},
        "compute_seconds": round(time.time() - started, 1),
    }
    os.makedirs(os.path.dirname(STRATEGY_FILE), exist_ok=True)
    with open(STRATEGY_FILE, "w") as fh:
        json.dump(strategy, fh, indent=2)

    print(f"Tick {tick} -> decision.json : {decision}")
    for p in ps:
        print(f"  {p}: signal {dec[p]:+.2f} -> après garde {decision[p]:+.3f}  ({reasons[p]})")
    print(f"  seuils bloc 1 : " + ", ".join(f"{p} {plan[0][p][0]:.0%}/{plan[0][p][1]:.0%}" for p in ps))
    print(f"  valeur prévue sur {L * nb} ticks : {value:.0f} (seuils fixes : {fixed_value:.0f}) -- {strategy['compute_seconds']} s")


if __name__ == "__main__":
    main()
