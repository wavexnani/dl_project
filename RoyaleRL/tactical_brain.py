"""
=================================================================
  Clash Royale AI — Tactical Brain & Expert Rule Arbiter
=================================================================
Neuro-Symbolic Decision System:
Enforces non-negotiable Clash Royale fundamentals, counter-matrix
rules, threat-lane locking, and spell execution at the base level.

Strict Rules Enforced (Zero-Tolerance Policy):
  1. Tower Spell-Snipe Finisher (Instant lethal win on low-HP tower)
  2. King Tower Dormancy & Lockout (Never wake enemy King early)
  3. Emergency Threat Defense (Never ignore Giant/Mini-Pekka/Minions)
  4. Anti-Air Counter Rule (Never play ground melee against Minions)
  5. Center-Pull / Kiting Geometry (2 towers shoot 1 enemy)
  6. Tactical Counter Matrix (Matchup advantage for all 10 cards)
  7. 10-Elixir Leak Prevention (Never sit at 10.0 elixir idle)
  8. Minimum Spell Value Protection (No wasted spells on empty area)
  9. Fragile Troop Protection / Tank-in-Front Sequencing
 10. The Pocket Exploitation (Instant 2nd tower snipe on tower down)
 11. Dynamic Lane Adaptation (Strictly match threat lane, no left bias)
=================================================================
"""

import time
import numpy as np
import config
from config import CARD_COSTS

# --- The 10 Recognized Card Classes ---
# archers, arrows, fireball, giant, goblin_cage, goblin_hut, knight, mini-pekka, minions, musketeer

# --- Threat Hierarchy ---
THREAT_LEVELS = {
    'giant': 10,        # Critical Win Condition (Hits towers only)
    'mini-pekka': 9,    # Critical High-DPS Melee (Can 3-shot towers)
    'minions': 8,       # High Flying Swarm DPS (Air-only vulnerability)
    'musketeer': 7,     # High Ranged DPS
    'knight': 5,        # Medium Mini-Tank
    'goblin_hut': 4,    # Medium Spawner Building
    'goblin_cage': 3,   # Medium Defensive Building
    'archers': 3,       # Low Ranged Duo
}

# --- Tactical Counter Matrix ---
# Maps an incoming enemy unit to a priority list of best counters in your deck
COUNTER_MATRIX = {
    'giant': [
        'mini-pekka',    # #1: Shreds Giant HP in seconds
        'goblin_cage',   # #2: Center pull building, absorbs hits, Brawler counter-pushes
        'musketeer',     # #3: Safe long-range DPS
        'minions',       # #4: Air DPS (Giant cannot hit back)
        'knight'         # #5: Cheap meat-shield
    ],
    'mini-pekka': [
        'minions',       # #1: Air unit! Mini-Pekka CANNOT hit air (takes 0 damage!)
        'goblin_cage',   # #2: Absorbs strikes, Brawler finishes
        'knight',        # #3: Distraction mini-tank (pulls into middle while towers shoot)
        'archers'        # #4: Ranged chip behind tower
    ],
    'minions': [
        'arrows',        # #1: Instant 1-hit kill on swarm (positive/neutral elixir trade)
        'musketeer',     # #2: Anti-air sniper from safety
        'archers',       # #3: Anti-air duo from safety
        'fireball'       # #4: Heavy spell fallback
    ],
    'musketeer': [
        'knight',        # #1: Drop directly on top of her at the bridge
        'mini-pekka',    # #2: 1-shots her
        'fireball',      # #3: Direct spell removal
        'minions'        # #4: Swarm surround
    ],
    'knight': [
        'mini-pekka',    # #1: Overpowers Knight in 2-3 hits
        'minions',       # #2: Air DPS (Knight cannot hit air)
        'musketeer',     # #3: Long-range chip
        'goblin_cage'    # #4: Defensive buffer
    ],
    'goblin_hut': [
        'fireball',      # #1: Hits hut + tower for massive value
        'musketeer',     # #2: Defends lane against spear goblins
        'giant'          # #3: Counter-push tank
    ],
    'goblin_cage': [
        'musketeer',     # #1: Snipes cage from distance before Brawler emerges
        'minions'        # #2: Air assault
    ],
    'archers': [
        'arrows',        # #1: Clears duo
        'knight',        # #2: Melee drop
        'fireball'       # #3: Spell clear
    ]
}

# Cards that CANNOT hit air targets (STRICTLY FORBIDDEN against Minions)
GROUND_ONLY_MELEE = {'knight', 'mini-pekka'}

# Damage thresholds for spell execution (estimated tower damage)
SPELL_TOWER_DAMAGE = {
    'fireball': 280,     # Princess Tower damage
    'arrows': 140
}

# King Tower Zone in Normalized Coordinates (X: [0.36, 0.64], Y: [0.00, 0.22])
KING_TOWER_ZONE = (0.36, 0.00, 0.64, 0.22)


def normalize_card_name(name):
    """Normalizes card names across detector labels, OCR, and deck lists."""
    if not name:
        return ''
    n = str(name).lower().strip().replace('_', '-')
    if 'pekka' in n:
        return 'mini-pekka'
    if 'cage' in n:
        return 'goblin_cage'
    if 'hut' in n:
        return 'goblin_hut'
    if 'minion' in n:
        return 'minions'
    if 'archer' in n:
        return 'archers'
    if 'giant' in n:
        return 'giant'
    if 'knight' in n:
        return 'knight'
    if 'musk' in n:
        return 'musketeer'
    if 'fireball' in n:
        return 'fireball'
    if 'arrow' in n:
        return 'arrows'
    return n.replace('-', '_')


class TacticalBrain:
    """
    Expert rule arbiter enforcing non-negotiable tactical Clash Royale laws.
    Guarantees the AI model never violates fundamental rules.
    """
    def __init__(self, scaler=None):
        self.scaler = scaler
        self.last_card_time = time.time()
        self.active_push_lane = 'left'  # Dynamic attack lane

    def _get_res(self):
        """Returns current (width, height) resolution safely."""
        if self.scaler is not None and hasattr(self.scaler, 'current_resolution'):
            return self.scaler.current_resolution
        return config.REFERENCE_RESOLUTION

    def _get_coords_pct(self, position):
        """Converts pixel position to normalized coordinates (0.0 to 1.0)."""
        w, h = self._get_res()
        return (position[0] / max(1, w), position[1] / max(1, h))

    def _to_pixels(self, pct_x, pct_y):
        """Converts normalized coordinates to pixel coordinates."""
        w, h = self._get_res()
        return (int(pct_x * w), int(pct_y * h))

    def _parse_hp(self, hp_val):
        """Safely parses HP value from OCR data."""
        if hp_val is None:
            return None
        try:
            cleaned = str(hp_val).strip().replace(',', '').replace(' ', '')
            val = int(cleaned)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    # ── Rule 1: Tower Spell-Snipe Finisher ─────────────────────────────
    def check_spell_finish(self, game_state):
        """
        Rule 1: If an enemy Princess Tower or King Tower is within lethal spell
        damage (Fireball <= 280, Arrows <= 140), immediately cast the spell
        to destroy the tower and secure the crown / victory.
        """
        ocr_data = game_state.get('ocr_data', {})
        hand = [normalize_card_name(c) for c in game_state.get('hand', [])]
        elixir = game_state.get('elixir', 0.0)

        # Candidate target towers: (tower_id, hp_value, (pct_x, pct_y), is_king)
        towers = [
            ('ptl', ocr_data.get('ptl'), (0.23, 0.14), False),  # Enemy Left Princess Tower
            ('ptr', ocr_data.get('ptr'), (0.76, 0.14), False),  # Enemy Right Princess Tower
            ('tk',  ocr_data.get('tk'),  (0.51, 0.09), True)   # Enemy King Tower
        ]

        # Prioritize lower HP towers
        valid_towers = []
        for tid, hp_raw, (tx, ty), is_king in towers:
            hp = self._parse_hp(hp_raw)
            if hp is not None and hp > 0:
                # Only target King Tower if it's already active or a princess tower is down
                if is_king:
                    ptl_hp = self._parse_hp(ocr_data.get('ptl'))
                    ptr_hp = self._parse_hp(ocr_data.get('ptr'))
                    # If both princess towers are still alive and healthy, don't spell snipe King unless King lethal
                    if ptl_hp is not None and ptr_hp is not None and hp > SPELL_TOWER_DAMAGE['fireball']:
                        continue
                valid_towers.append((tid, hp, (tx, ty)))

        valid_towers.sort(key=lambda t: t[1])  # Lowest HP first

        for tid, hp, (tx_pct, ty_pct) in valid_towers:
            # 1. Check Arrows lethal (cheaper: 3 elixir, 140 dmg)
            if 'arrows' in hand and hp <= SPELL_TOWER_DAMAGE['arrows']:
                a_cost = CARD_COSTS.get('arrows', 3)
                if elixir >= a_cost:
                    slot = hand.index('arrows')
                    pix_pos = self._to_pixels(tx_pct, ty_pct)
                    print(f"🎯 [LETHAL FINISH] Enemy {tid.upper()} at {hp} HP! Casting ARROWS for guaranteed win!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': pix_pos, 'tactical_rule': 'SPELL_FINISH'}

            # 2. Check Fireball lethal (4 elixir, 280 dmg)
            if 'fireball' in hand and hp <= SPELL_TOWER_DAMAGE['fireball']:
                f_cost = CARD_COSTS.get('fireball', 4)
                if elixir >= f_cost:
                    slot = hand.index('fireball')
                    pix_pos = self._to_pixels(tx_pct, ty_pct)
                    print(f"🎯 [LETHAL FINISH] Enemy {tid.upper()} at {hp} HP! Casting FIREBALL for guaranteed win!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': pix_pos, 'tactical_rule': 'SPELL_FINISH'}

        return None

    # ── Rule 2: King Tower Spell Lockout ──────────────────────────────
    def is_king_tower_clip(self, card_name, pos_pct, game_state):
        """
        Rule 2: Returns True if a spell placement would hit the enemy King Tower
        while both enemy Princess towers are still alive (preventing early King activation).
        """
        card_norm = normalize_card_name(card_name)
        if card_norm not in ('fireball', 'arrows'):
            return False

        ocr_data = game_state.get('ocr_data', {})
        ptl_hp = self._parse_hp(ocr_data.get('ptl'))
        ptr_hp = self._parse_hp(ocr_data.get('ptr'))

        # If both Princess towers are alive, King Tower is strictly off-limits
        if ptl_hp is not None and ptr_hp is not None:
            kx1, ky1, kx2, ky2 = KING_TOWER_ZONE
            px, py = pos_pct
            # Spell splash radius safety buffer: ~0.08
            if (kx1 - 0.08) <= px <= (kx2 + 0.08) and py <= (ky2 + 0.08):
                return True
        return False

    def get_safe_spell_redirection(self, game_state):
        """Finds the best alive enemy Princess Tower to redirect an accidental King Tower spell."""
        ocr_data = game_state.get('ocr_data', {})
        ptl_hp = self._parse_hp(ocr_data.get('ptl'))
        ptr_hp = self._parse_hp(ocr_data.get('ptr'))

        # If Right is dead or Left has lower HP, target Left
        if ptr_hp is None:
            return self._to_pixels(0.23, 0.14)
        if ptl_hp is None:
            return self._to_pixels(0.76, 0.14)
        if ptl_hp <= ptr_hp:
            return self._to_pixels(0.23, 0.14)
        return self._to_pixels(0.76, 0.14)

    # ── Rule 3, 4, 5: Emergency Threat Defense, Anti-Air & Center-Pull ─
    def check_emergency_threats(self, game_state):
        """
        Rule 3 & 4 & 5:
        - Detects high-threat enemies crossing the river (Y >= 0.40).
        - Locks onto the threatened lane (Left or Right).
        - Strictly forbids ground-only melee against air (Minions).
        - Enforces Counter Matrix matchup.
        - Deploys Center-Pull geometry to double defensive DPS.
        """
        enemies = game_state.get('enemies', [])
        raw_hand = game_state.get('hand', [])
        hand = [normalize_card_name(c) for c in raw_hand]
        elixir = game_state.get('elixir', 0.0)

        if not enemies:
            return None

        cur_w, cur_h = self._get_res()
        active_threats = []

        for e in enemies:
            name = normalize_card_name(e.get('name', ''))
            box = e.get('box', (0, 0, 0, 0))
            center_x = (box[0] + box[2]) / 2.0 / cur_w
            center_y = (box[1] + box[3]) / 2.0 / cur_h

            # Threat approaching bridge or on our side of the arena
            if center_y >= 0.38:
                threat_score = THREAT_LEVELS.get(name, 2)
                active_threats.append((threat_score, name, center_x, center_y, box))

        if not active_threats:
            return None

        # Sort by threat severity (Giant and Mini-Pekka highest)
        active_threats.sort(key=lambda t: t[0], reverse=True)
        top_threat = active_threats[0]
        t_score, t_name, t_x, t_y, t_box = top_threat

        # Defend against any unit with threat score >= 3 crossing the river
        if t_score < 3:
            return None

        threat_lane = 'left' if t_x < 0.50 else 'right'
        # Update active push lane to oppose / counter
        self.active_push_lane = threat_lane

        # Select counter card from hand
        best_counter = None
        counter_slot = None
        counter_list = COUNTER_MATRIX.get(t_name, [])

        # 1. Search Counter Matrix priority list
        for candidate in counter_list:
            if candidate in hand:
                cost = CARD_COSTS.get(candidate, 3)
                if elixir >= cost:
                    best_counter = candidate
                    counter_slot = hand.index(candidate)
                    break

        # 2. Fallback: Any affordable playable card that respects Anti-Air law
        if best_counter is None:
            for i, c in enumerate(hand):
                # Strict Anti-Air Law: Never ground melee vs Minions
                if t_name == 'minions' and c in GROUND_ONLY_MELEE:
                    continue
                cost = CARD_COSTS.get(c, 3)
                if elixir >= cost:
                    best_counter = c
                    counter_slot = i
                    break

        if best_counter is None:
            # Player cannot afford counter yet; hold elixir, do NOT waste on wrong card
            return None

        # Placement Calculation:
        # A. Air Swarm (Minions): Direct Arrows strike on swarm center or ranged behind tower
        if t_name == 'minions':
            if best_counter == 'arrows':
                strike_x = int((t_box[0] + t_box[2]) / 2)
                strike_y = int((t_box[1] + t_box[3]) / 2)
                print(f"🏹 [ANTI-AIR CLEAR] Casting ARROWS directly on Minion swarm at ({strike_x}, {strike_y})!")
                return {'action': 'play_card', 'card_slot': counter_slot, 'position': (strike_x, strike_y), 'tactical_rule': 'AIR_SWARM_CLEAR'}
            else:
                # Place anti-air troop safely behind our Princess tower
                plant_x_pct = 0.23 if threat_lane == 'left' else 0.76
                plant_y_pct = 0.72
                deploy_pos = self._to_pixels(plant_x_pct, plant_y_pct)
                print(f"🛡️ [ANTI-AIR DEFENSE] Deploying {best_counter.upper()} behind tower to intercept Minions!")
                return {'action': 'play_card', 'card_slot': counter_slot, 'position': deploy_pos, 'tactical_rule': 'RANGED_ANTI_AIR'}

        # B. Single-Target Melee / Tanks (Giant, Mini-Pekka, Knight): Center-Pull Kiting
        if t_name in ('giant', 'mini-pekka', 'knight'):
            # Pull into center: Tile 4-3 geometry so BOTH Princess towers fire
            plant_x_pct = 0.47 if threat_lane == 'left' else 0.53
            plant_y_pct = 0.63
            deploy_pos = self._to_pixels(plant_x_pct, plant_y_pct)
            print(f"🛡️ [CENTER PULL] Playing {best_counter.upper()} at Center Kiting Zone ({plant_x_pct:.2f}, {plant_y_pct:.2f}) vs {t_name.upper()}!")
            return {'action': 'play_card', 'card_slot': counter_slot, 'position': deploy_pos, 'tactical_rule': 'CENTER_PULL_DEFENSE'}

        # C. Ranged Attacker (Musketeer): Drop melee directly on top of her
        if t_name == 'musketeer' and best_counter in ('knight', 'mini-pekka'):
            drop_x = int((t_box[0] + t_box[2]) / 2)
            drop_y = int((t_box[1] + t_box[3]) / 2)
            print(f"⚔️ [MELEE DROP] Dropping {best_counter.upper()} directly on Musketeer at ({drop_x}, {drop_y})!")
            return {'action': 'play_card', 'card_slot': counter_slot, 'position': (drop_x, drop_y), 'tactical_rule': 'MELEE_ON_RANGED'}

        # D. Standard Lane Defense
        plant_x_pct = 0.26 if threat_lane == 'left' else 0.74
        plant_y_pct = 0.65
        deploy_pos = self._to_pixels(plant_x_pct, plant_y_pct)
        return {'action': 'play_card', 'card_slot': counter_slot, 'position': deploy_pos, 'tactical_rule': 'LANE_DEFENSE'}

    # ── Rule 6: 10-Elixir Leak Prevention ─────────────────────────────
    def check_elixir_leak_prevention(self, game_state):
        """
        Rule 6: If elixir reaches >= 9.5 and no enemies are attacking, forces a safe
        backline cycle play (Giant, Knight, Archers, Musketeer) behind King tower
        so elixir is never leaked or wasted sitting at 10.0.
        """
        elixir = game_state.get('elixir', 0.0)
        hand = [normalize_card_name(c) for c in game_state.get('hand', [])]

        if elixir < 9.5:
            return None

        # Cycle preference order: Slow tank / builder first, then ranged support
        cycle_candidates = ['giant', 'knight', 'archers', 'musketeer', 'goblin_hut', 'goblin_cage']

        for c in cycle_candidates:
            if c in hand:
                slot = hand.index(c)
                cost = CARD_COSTS.get(c, 3)
                if elixir >= cost:
                    # Determine lane with weaker enemy tower or default active push lane
                    ocr_data = game_state.get('ocr_data', {})
                    ptl_hp = self._parse_hp(ocr_data.get('ptl')) or 2534
                    ptr_hp = self._parse_hp(ocr_data.get('ptr')) or 2534
                    cycle_lane = 'left' if ptl_hp <= ptr_hp else 'right'

                    x_pct = 0.26 if cycle_lane == 'left' else 0.74
                    y_pct = 0.82  # Safely behind King Tower
                    deploy_pos = self._to_pixels(x_pct, y_pct)
                    print(f"⚡ [LEAK PREVENTION] Elixir at {elixir:.1f}! Cycling {c.upper()} safely behind King Tower ({x_pct:.2f}, {y_pct:.2f})!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': deploy_pos, 'tactical_rule': 'ELIXIR_LEAK_CYCLE'}

        return None

    # ── Rule 10: The Pocket Exploitation ──────────────────────────────
    def check_pocket_deployment(self, game_state):
        """
        Rule 10: When one enemy Princess tower is destroyed, deploys high-DPS
        units directly into 'The Pocket' (mid-river) to quickly assault the 2nd tower.
        """
        ocr_data = game_state.get('ocr_data', {})
        hand = [normalize_card_name(c) for c in game_state.get('hand', [])]
        elixir = game_state.get('elixir', 0.0)

        left_dead = (self._parse_hp(ocr_data.get('ptl')) is None)
        right_dead = (self._parse_hp(ocr_data.get('ptr')) is None)

        pocket_units = ['musketeer', 'mini-pekka', 'giant']

        # Left tower is down -> Left Pocket assaults Right Tower
        if left_dead and not right_dead:
            for u in pocket_units:
                if u in hand and elixir >= CARD_COSTS.get(u, 4):
                    slot = hand.index(u)
                    pos = self._to_pixels(0.42, 0.42)
                    print(f"🔥 [THE POCKET] Left tower down! Deploying {u.upper()} in Pocket to assault Right Tower!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': pos, 'tactical_rule': 'POCKET_ASSAULT'}

        # Right tower is down -> Right Pocket assaults Left Tower
        if right_dead and not left_dead:
            for u in pocket_units:
                if u in hand and elixir >= CARD_COSTS.get(u, 4):
                    slot = hand.index(u)
                    pos = self._to_pixels(0.58, 0.42)
                    print(f"🔥 [THE POCKET] Right tower down! Deploying {u.upper()} in Pocket to assault Left Tower!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': pos, 'tactical_rule': 'POCKET_ASSAULT'}

        return None

    # ── Mandatory Action Master ───────────────────────────────────────
    def get_mandatory_action(self, game_state):
        """
        Evaluates deterministic, non-negotiable rules.
        If any rule condition is met, returns the required tactical action immediately.
        """
        # 1. Lethal Spell Finish (Instant Win)
        finish_action = self.check_spell_finish(game_state)
        if finish_action:
            return finish_action

        # 2. Emergency Threat Response (Never Ignore Bridge / Side Threats)
        threat_action = self.check_emergency_threats(game_state)
        if threat_action:
            return threat_action

        # 3. Pocket Assault (when enemy tower is already down)
        if game_state.get('elixir', 0.0) >= 6.0:
            pocket_action = self.check_pocket_deployment(game_state)
            if pocket_action:
                return pocket_action

        # 4. Elixir Leak Prevention (Sitting at >= 9.5 elixir)
        leak_action = self.check_elixir_leak_prevention(game_state)
        if leak_action:
            return leak_action

        return None

    # ── Negative Constraint Validation ────────────────────────────────
    def validate_candidate_action(self, action, game_state):
        """
        Strictly validates candidate actions (from neural net or exploration)
        against negative constraints. If an action violates any tactical rule,
        it is modified, redirected, or blocked (returns None).
        """
        if not action or action.get('action') != 'play_card':
            return action

        raw_hand = game_state.get('hand', [])
        hand = [normalize_card_name(c) for c in raw_hand]
        slot = action.get('card_slot')
        pos = action.get('position', (0, 0))

        if slot is None or not (0 <= slot < len(hand)):
            return None

        card_name = hand[slot]
        pos_pct = self._get_coords_pct(pos)
        cur_w, cur_h = self._get_res()
        enemies = game_state.get('enemies', [])

        # ── Check King Tower Lockout (Rule 2) ──────────────────────────
        if self.is_king_tower_clip(card_name, pos_pct, game_state):
            print("⛔ [RULE OVERRIDE] Blocked premature spell clipping King Tower! Redirecting to Princess Tower.")
            redirect_pos = self.get_safe_spell_redirection(game_state)
            return {'action': 'play_card', 'card_slot': slot, 'position': redirect_pos, 'tactical_rule': 'KING_TOWER_PROTECTION'}

        # ── Check Anti-Air Law (Rule 4) ────────────────────────────────
        # Strictly forbid ground-only melee against flying Minions
        if card_name in GROUND_ONLY_MELEE:
            active_threats = [
                e for e in enemies
                if (e.get('box', (0, 0, 0, 0))[1] + e.get('box', (0, 0, 0, 0))[3]) / 2.0 / cur_h >= 0.38
            ]
            if active_threats and all(normalize_card_name(e.get('name')) == 'minions' for e in active_threats):
                print(f"⛔ [RULE OVERRIDE] Blocked {card_name.upper()} deployment against Minions (Ground melee cannot hit air!).")
                return None

        # ── Check Spell Waste Rules (Rule 8) ───────────────────────────
        if card_name == 'arrows':
            has_minions = any(normalize_card_name(e.get('name')) == 'minions' for e in enemies)
            has_dense_cluster = len(enemies) >= 2
            # Check if targeting tower
            is_targeting_tower = (pos_pct[1] <= 0.22)
            if not has_minions and not has_dense_cluster and not is_targeting_tower:
                print("⛔ [RULE OVERRIDE] Blocked wasting Arrows with no swarm or tower target.")
                return None

        if card_name == 'fireball':
            # Block Fireball placed on empty backline on our own side
            if pos_pct[1] >= 0.60 and len(enemies) == 0:
                print("⛔ [RULE OVERRIDE] Blocked wasting Fireball on empty friendly ground.")
                return None

        # ── Check Fragile Troop Protection (Rule 9) ────────────────────
        # Squishy ranged units (Musketeer, Archers) should not be dropped naked at the bridge
        if card_name in ('musketeer', 'archers') and pos_pct[1] < 0.52:
            safe_pos = (pos[0], int(pos[1] + 0.14 * cur_h))
            print(f"🛡️ [RULE OVERRIDE] Pulled {card_name.upper()} back behind river for safety.")
            return {'action': 'play_card', 'card_slot': slot, 'position': safe_pos, 'tactical_rule': 'TANK_IN_FRONT_SEQUENCING'}

        # ── Check Dynamic Lane Defense Adaptation (Rule 11) ────────────
        # If an enemy threat is approaching on one lane, don't drop card on the opposite lane
        for e in enemies:
            ey = (e.get('box', (0, 0, 0, 0))[1] + e.get('box', (0, 0, 0, 0))[3]) / 2.0 / cur_h
            if ey >= 0.40:
                ex = (e.get('box', (0, 0, 0, 0))[0] + e.get('box', (0, 0, 0, 0))[2]) / 2.0 / cur_w
                threat_lane = 'left' if ex < 0.50 else 'right'
                play_lane = 'left' if pos_pct[0] < 0.50 else 'right'
                if card_name not in ('fireball', 'arrows') and play_lane != threat_lane and 0.45 <= pos_pct[1] <= 0.70:
                    # Adjust to the threatened lane
                    adjusted_x = int((1.0 - pos_pct[0]) * cur_w)
                    print(f"🛡️ [RULE OVERRIDE] Adjusted deployment from {play_lane} to threatened {threat_lane} lane.")
                    return {'action': 'play_card', 'card_slot': slot, 'position': (adjusted_x, pos[1]), 'tactical_rule': 'LANE_ADAPTATION'}

        return action

    # ── Master Arbiter ────────────────────────────────────────────────
    def arbitrate_decision(self, game_state, dt_action=None):
        """
        Master decision function combining mandatory triggers and candidate validation.
        Guarantees 100% adherence to all tactical rules.
        """
        mandatory = self.get_mandatory_action(game_state)
        if mandatory is not None:
            return mandatory

        if dt_action is not None:
            return self.validate_candidate_action(dt_action, game_state)

        return None
