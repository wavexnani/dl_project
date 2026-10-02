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
 12. Tank Bridge Drop Prohibition (NEVER drop Giant naked at the bridge)
 13. Threat Memory & Persistence (No momentary blindness drops defense)
 14. Tower Damage Sensor (Instant defense trigger on tower HP drop)
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

        # Threat Persistence Memory: preserves threats across frames so detector flicker doesn't lose threats
        self.active_threat_memory = []

        # Tower Damage Sensor: tracks our Princess towers to detect attacks instantly
        self.last_pbl_hp = None
        self.last_pbr_hp = None

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

    def get_active_threat_lane(self):
        """Returns the lane currently under threat from active threat memory, or None."""
        now = time.time()
        valid = [t for t in self.active_threat_memory if (now - t['timestamp']) < 4.5]
        if valid:
            valid.sort(key=lambda t: t['score'], reverse=True)
            return valid[0]['lane']
        return None

    def calculate_predictive_aim(self, box, enemy_name='minions', spell_name='arrows'):
        """
        Calculates the lead intercept point for spells (Arrows, Fireball) targeting moving enemies.
        Accounts for spell deploy delay (1.0s) + distance-based projectile flight time.
        Applies distance-decay clamping near the Princess tower to prevent over-shooting.
        """
        cur_w, cur_h = self._get_res()
        cx_pct = ((box[0] + box[2]) / 2.0) / cur_w
        cy_pct = ((box[1] + box[3]) / 2.0) / cur_h

        # Destination Princess Tower
        target_tower_x = 0.23 if cx_pct < 0.50 else 0.76
        target_tower_y = 0.62

        # Clash Royale troop speeds (normalized screen Y per second)
        speeds = {
            'minions': 0.048,      # Fast (84 tiles/min)
            'mini-pekka': 0.048,   # Fast
            'knight': 0.034,       # Medium
            'musketeer': 0.034,    # Medium
            'archers': 0.034,      # Medium
            'giant': 0.022         # Slow
        }
        v = speeds.get(enemy_name.lower(), 0.035)

        # Distance from our King Tower (0.50, 0.76) to current target position
        dist = np.hypot(cx_pct - 0.50, cy_pct - 0.76)
        flight_time = 1.0 + (dist / 0.65)  # 1.0s deploy + projectile flight

        # Stop-Zone Clamping: If enemy is near the Princess tower (Y >= 0.54),
        # they are locking onto or attacking the tower, so decay forward lead to 0!
        if cy_pct >= 0.58:
            lead_factor = 0.0
        elif cy_pct >= 0.44:
            lead_factor = max(0.0, (0.58 - cy_pct) / (0.58 - 0.44))
        else:
            lead_factor = 1.0

        # Projected impact coordinates
        lead_y_pct = cy_pct + (v * flight_time * lead_factor)
        lead_y_pct = min(0.60, max(cy_pct, lead_y_pct))

        # Interpolate X along the vector towards the target Princess tower
        if target_tower_y > cy_pct:
            progress = (lead_y_pct - cy_pct) / (target_tower_y - cy_pct)
            lead_x_pct = cx_pct + progress * (target_tower_x - cx_pct)
        else:
            lead_x_pct = cx_pct

        pixel_x = int(lead_x_pct * cur_w)
        pixel_y = int(lead_y_pct * cur_h)
        print(f"🎯 [PREDICTIVE LEAD] Leading {enemy_name.upper()} ({cx_pct:.2f}, {cy_pct:.2f}) -> Intercept ({lead_x_pct:.2f}, {lead_y_pct:.2f})")
        return (pixel_x, pixel_y)

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
            ('ptl', ocr_data.get('ptl'), (0.20, 0.14), False),  # Enemy Left Princess Tower (calibrated safe outward)
            ('ptr', ocr_data.get('ptr'), (0.79, 0.14), False),  # Enemy Right Princess Tower (calibrated safe outward)
            ('tk',  ocr_data.get('tk'),  (0.51, 0.09), True)   # Enemy King Tower
        ]

        valid_towers = []
        for tid, hp_raw, (tx, ty), is_king in towers:
            hp = self._parse_hp(hp_raw)
            if hp is not None and hp > 0:
                if is_king:
                    ptl_hp = self._parse_hp(ocr_data.get('ptl'))
                    ptr_hp = self._parse_hp(ocr_data.get('ptr'))
                    if ptl_hp is not None and ptr_hp is not None and hp > SPELL_TOWER_DAMAGE['fireball']:
                        continue
                valid_towers.append((tid, hp, (tx, ty)))

        valid_towers.sort(key=lambda t: t[1])  # Lowest HP first

        for tid, hp, (tx_pct, ty_pct) in valid_towers:
            if 'arrows' in hand and hp <= SPELL_TOWER_DAMAGE['arrows']:
                a_cost = CARD_COSTS.get('arrows', 3)
                if elixir >= a_cost:
                    slot = hand.index('arrows')
                    pix_pos = self._to_pixels(tx_pct, ty_pct)
                    print(f"🎯 [LETHAL FINISH] Enemy {tid.upper()} at {hp} HP! Casting ARROWS for guaranteed win!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': pix_pos, 'tactical_rule': 'SPELL_FINISH'}

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
        card_norm = normalize_card_name(card_name)
        if card_norm not in ('fireball', 'arrows'):
            return False

        ocr_data = game_state.get('ocr_data', {})
        ptl_hp = self._parse_hp(ocr_data.get('ptl'))
        ptr_hp = self._parse_hp(ocr_data.get('ptr'))

        if ptl_hp is not None and ptr_hp is not None:
            kx1, ky1, kx2, ky2 = KING_TOWER_ZONE
            px, py = pos_pct
            if (kx1 - 0.08) <= px <= (kx2 + 0.08) and py <= (ky2 + 0.08):
                return True
        return False

    def get_safe_spell_redirection(self, game_state):
        ocr_data = game_state.get('ocr_data', {})
        ptl_hp = self._parse_hp(ocr_data.get('ptl'))
        ptr_hp = self._parse_hp(ocr_data.get('ptr'))

        if ptr_hp is None:
            return self._to_pixels(0.23, 0.14)
        if ptl_hp is None:
            return self._to_pixels(0.76, 0.14)
        if ptl_hp <= ptr_hp:
            return self._to_pixels(0.23, 0.14)
        return self._to_pixels(0.76, 0.14)

    # ── Rule 3, 4, 5, 13, 14: Emergency Threat Defense & Memory ───────
    def check_emergency_threats(self, game_state):
        """
        Rule 3, 4, 5, 13, 14:
        - Detects high-threat enemies crossing the river (Y >= 0.38).
        - Uses Threat Memory to persist threats across frames (no flicker blindness).
        - Uses Tower Damage Sensor to instantly detect attacks if our tower HP drops.
        - Deploys hard counters using Center-Pull geometry.
        """
        now = time.time()
        enemies = game_state.get('enemies', [])
        raw_hand = game_state.get('hand', [])
        hand = [normalize_card_name(c) for c in raw_hand]
        elixir = game_state.get('elixir', 0.0)
        cur_w, cur_h = self._get_res()

        # 1. Clean expired threats (> 4.5 seconds old)
        self.active_threat_memory = [t for t in self.active_threat_memory if (now - t['timestamp']) < 4.5]

        # 2. Ingest active vision detections
        for e in enemies:
            name = normalize_card_name(e.get('name', ''))
            box = e.get('box', (0, 0, 0, 0))
            center_x = (box[0] + box[2]) / 2.0 / cur_w
            center_y = (box[1] + box[3]) / 2.0 / cur_h

            # Approaching bridge or on our side of the arena
            if center_y >= 0.36:
                threat_score = THREAT_LEVELS.get(name, 2)
                if threat_score >= 3:
                    threat_lane = 'left' if center_x < 0.50 else 'right'
                    # Update or add in memory
                    self.active_threat_memory = [t for t in self.active_threat_memory if t['name'] != name]
                    self.active_threat_memory.append({
                        'name': name,
                        'score': threat_score,
                        'lane': threat_lane,
                        'x': center_x,
                        'y': center_y,
                        'box': box,
                        'timestamp': now
                    })

        # 3. Tower Damage Sensor (OCR HP drop on friendly towers)
        ocr_data = game_state.get('ocr_data', {})
        pbl_hp = self._parse_hp(ocr_data.get('pbl'))
        pbr_hp = self._parse_hp(ocr_data.get('pbr'))

        if self.last_pbl_hp is not None and pbl_hp is not None and (self.last_pbl_hp - pbl_hp) >= 30:
            print(f"🚨 [TOWER SENSOR] Left Tower damaged ({self.last_pbl_hp} -> {pbl_hp})! Locking emergency defense on LEFT lane!")
            self.active_threat_memory.append({
                'name': 'giant',
                'score': 10,
                'lane': 'left',
                'x': 0.25,
                'y': 0.60,
                'box': (int(0.20 * cur_w), int(0.55 * cur_h), int(0.30 * cur_w), int(0.65 * cur_h)),
                'timestamp': now
            })
        elif self.last_pbr_hp is not None and pbr_hp is not None and (self.last_pbr_hp - pbr_hp) >= 30:
            print(f"🚨 [TOWER SENSOR] Right Tower damaged ({self.last_pbr_hp} -> {pbr_hp})! Locking emergency defense on RIGHT lane!")
            self.active_threat_memory.append({
                'name': 'giant',
                'score': 10,
                'lane': 'right',
                'x': 0.75,
                'y': 0.60,
                'box': (int(0.70 * cur_w), int(0.55 * cur_h), int(0.80 * cur_w), int(0.65 * cur_h)),
                'timestamp': now
            })

        if pbl_hp is not None:
            self.last_pbl_hp = pbl_hp
        if pbr_hp is not None:
            self.last_pbr_hp = pbr_hp

        if not self.active_threat_memory:
            return None

        # Sort threats by severity
        self.active_threat_memory.sort(key=lambda t: t['score'], reverse=True)
        top_threat = self.active_threat_memory[0]
        t_name = top_threat['name']
        threat_lane = top_threat['lane']
        t_box = top_threat['box']
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
                if t_name == 'minions' and c in GROUND_ONLY_MELEE:
                    continue
                cost = CARD_COSTS.get(c, 3)
                if elixir >= cost:
                    best_counter = c
                    counter_slot = i
                    break

        if best_counter is None:
            # Need more elixir to deploy counter; hold elixir, do NOT spend on wrong card
            return None

        # Placement Calculation:
        # A. Air Swarm (Minions): Direct Arrows strike with predictive lead aiming
        if t_name == 'minions':
            if best_counter == 'arrows':
                strike_pos = self.calculate_predictive_aim(t_box, enemy_name='minions', spell_name='arrows')
                print(f"🏹 [ANTI-AIR CLEAR] Casting ARROWS with predictive lead aiming at {strike_pos}!")
                return {'action': 'play_card', 'card_slot': counter_slot, 'position': strike_pos, 'tactical_rule': 'AIR_SWARM_CLEAR'}
            else:
                plant_x_pct = 0.23 if threat_lane == 'left' else 0.76
                plant_y_pct = 0.72
                deploy_pos = self._to_pixels(plant_x_pct, plant_y_pct)
                print(f"🛡️ [ANTI-AIR DEFENSE] Deploying {best_counter.upper()} behind tower to intercept Minions!")
                return {'action': 'play_card', 'card_slot': counter_slot, 'position': deploy_pos, 'tactical_rule': 'RANGED_ANTI_AIR'}

        # B. Single-Target Melee / Tanks (Giant, Mini-Pekka, Knight): Center-Pull Kiting
        if t_name in ('giant', 'mini-pekka', 'knight'):
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
    def check_elixir_leak_prevention(self, game_state, min_elixir=9.5):
        elixir = game_state.get('elixir', 0.0)
        hand = [normalize_card_name(c) for c in game_state.get('hand', [])]

        if elixir < min_elixir:
            return None

        cycle_candidates = ['giant', 'knight', 'archers', 'musketeer', 'goblin_hut', 'goblin_cage', 'mini-pekka', 'minions']

        for c in cycle_candidates:
            if c in hand:
                slot = hand.index(c)
                cost = CARD_COSTS.get(c, 3)
                if elixir >= cost:
                    ocr_data = game_state.get('ocr_data', {})
                    ptl_hp = self._parse_hp(ocr_data.get('ptl')) or 2534
                    ptr_hp = self._parse_hp(ocr_data.get('ptr')) or 2534
                    cycle_lane = 'left' if ptl_hp <= ptr_hp else 'right'

                    x_pct = 0.26 if cycle_lane == 'left' else 0.74
                    y_pct = 0.76  # Safely behind King Tower on arena grass (well above card UI)
                    deploy_pos = self._to_pixels(x_pct, y_pct)
                    print(f"⚡ [LEAK PREVENTION] Elixir at {elixir:.1f}! Cycling {c.upper()} safely behind King Tower ({x_pct:.2f}, {y_pct:.2f})!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': deploy_pos, 'tactical_rule': 'ELIXIR_LEAK_CYCLE'}

        return None

    # ── Rule 10: The Pocket Exploitation ──────────────────────────────
    def check_pocket_deployment(self, game_state):
        ocr_data = game_state.get('ocr_data', {})
        hand = [normalize_card_name(c) for c in game_state.get('hand', [])]
        elixir = game_state.get('elixir', 0.0)

        left_dead = (self._parse_hp(ocr_data.get('ptl')) is None)
        right_dead = (self._parse_hp(ocr_data.get('ptr')) is None)

        pocket_units = ['musketeer', 'mini-pekka', 'giant']

        if left_dead and not right_dead:
            for u in pocket_units:
                if u in hand and elixir >= CARD_COSTS.get(u, 4):
                    slot = hand.index(u)
                    pos = self._to_pixels(0.42, 0.42)
                    print(f"🔥 [THE POCKET] Left tower down! Deploying {u.upper()} in Pocket to assault Right Tower!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': pos, 'tactical_rule': 'POCKET_ASSAULT'}

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
        finish_action = self.check_spell_finish(game_state)
        if finish_action:
            return finish_action

        threat_action = self.check_emergency_threats(game_state)
        if threat_action:
            return threat_action

        if game_state.get('elixir', 0.0) >= 6.0:
            pocket_action = self.check_pocket_deployment(game_state)
            if pocket_action:
                return pocket_action

        leak_action = self.check_elixir_leak_prevention(game_state)
        if leak_action:
            return leak_action

        return None

    # ── Negative Constraint Validation ────────────────────────────────
    def validate_candidate_action(self, action, game_state):
        """
        Strictly validates candidate actions against non-negotiable negative constraints:
        - Blocks premature King Tower hits
        - Blocks ground melee vs flying
        - Blocks bridge Giant drops
        - Blocks attacking enemy territory while defending
        - Blocks spell waste
        - Pulls fragile troops behind river
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

        # Ingest current frame enemies into memory if not already present
        now = time.time()
        for e in enemies:
            name = normalize_card_name(e.get('name', ''))
            box = e.get('box', (0, 0, 0, 0))
            center_x = (box[0] + box[2]) / 2.0 / cur_w
            center_y = (box[1] + box[3]) / 2.0 / cur_h
            if center_y >= 0.36:
                threat_score = THREAT_LEVELS.get(name, 2)
                if threat_score >= 3:
                    t_lane = 'left' if center_x < 0.50 else 'right'
                    if not any(t['name'] == name for t in self.active_threat_memory):
                        self.active_threat_memory.append({
                            'name': name, 'score': threat_score, 'lane': t_lane,
                            'x': center_x, 'y': center_y, 'box': box, 'timestamp': now
                        })

        active_threat_lane = self.get_active_threat_lane()

        # ── Check King Tower Lockout (Rule 2) ──────────────────────────
        if self.is_king_tower_clip(card_name, pos_pct, game_state):
            print("⛔ [RULE OVERRIDE] Blocked premature spell clipping King Tower! Redirecting to Princess Tower.")
            redirect_pos = self.get_safe_spell_redirection(game_state)
            return {'action': 'play_card', 'card_slot': slot, 'position': redirect_pos, 'tactical_rule': 'KING_TOWER_PROTECTION'}

        # ── Check Anti-Air Law (Rule 4) ────────────────────────────────
        if card_name in GROUND_ONLY_MELEE:
            has_air_threat = any(
                normalize_card_name(e.get('name')) == 'minions'
                for e in enemies
                if (e.get('box', (0, 0, 0, 0))[1] + e.get('box', (0, 0, 0, 0))[3]) / 2.0 / cur_h >= 0.36
            ) or any(t['name'] == 'minions' for t in self.active_threat_memory)
            if has_air_threat:
                print(f"⛔ [RULE OVERRIDE] Blocked {card_name.upper()} deployment against Minions (Ground melee cannot hit air!).")
                return None

        # ── Check Tank Bridge Drop Prohibition (Rule 12) ───────────────
        # Giant must NEVER be dropped aggressively at the bridge (pos_pct[1] < 0.55)
        if card_name == 'giant' and pos_pct[1] < 0.55:
            if active_threat_lane:
                pull_x = 0.47 if active_threat_lane == 'left' else 0.53
                safe_pos = self._to_pixels(pull_x, 0.63)
                print(f"⛔ [RULE OVERRIDE] Blocked Bridge Giant! Redirected to Center-Pull Defense ({pull_x:.2f}, 0.63).")
                return {'action': 'play_card', 'card_slot': slot, 'position': safe_pos, 'tactical_rule': 'CENTER_PULL_DEFENSE'}
            else:
                cycle_lane = 'left' if pos_pct[0] < 0.50 else 'right'
                back_x = 0.26 if cycle_lane == 'left' else 0.74
                safe_pos = self._to_pixels(back_x, 0.76)
                print(f"⛔ [RULE OVERRIDE] Blocked Bridge Giant! Redirected to safe backline deployment ({back_x:.2f}, 0.76).")
                return {'action': 'play_card', 'card_slot': slot, 'position': safe_pos, 'tactical_rule': 'SAFE_BACKLINE_TANK'}

        # ── Check No Attack During Active Defense (Rule 15) ─────────────
        # If any threat is attacking on our side, NEVER play into enemy territory
        if active_threat_lane and pos_pct[1] < 0.50 and card_name not in ('fireball', 'arrows'):
            print(f"⛔ [RULE OVERRIDE] Blocked offensive play on enemy side during active {active_threat_lane.upper()} defense! Redirecting.")
            def_pos = self._to_pixels(0.47 if active_threat_lane == 'left' else 0.53, 0.63)
            return {'action': 'play_card', 'card_slot': slot, 'position': def_pos, 'tactical_rule': 'DEFENSIVE_LANE_LOCK'}

        # ── Check Spell Waste Rules (Rule 8) ───────────────────────────
        if card_name == 'arrows':
            has_minions = any(normalize_card_name(e.get('name')) == 'minions' for e in enemies)
            has_dense_cluster = len(enemies) >= 2
            is_targeting_tower = (pos_pct[1] <= 0.22)
            if not has_minions and not has_dense_cluster and not is_targeting_tower:
                print("⛔ [RULE OVERRIDE] Blocked wasting Arrows with no swarm or tower target.")
                return None

        if card_name == 'fireball':
            if pos_pct[1] >= 0.60 and len(enemies) == 0:
                print("⛔ [RULE OVERRIDE] Blocked wasting Fireball on empty friendly ground.")
                return None

        # ── Check Fragile Troop Protection (Rule 9) ────────────────────
        if card_name in ('musketeer', 'archers') and pos_pct[1] < 0.52:
            safe_pos = (pos[0], int(pos[1] + 0.14 * cur_h))
            print(f"🛡️ [RULE OVERRIDE] Pulled {card_name.upper()} back behind river for safety.")
            return {'action': 'play_card', 'card_slot': slot, 'position': safe_pos, 'tactical_rule': 'TANK_IN_FRONT_SEQUENCING'}

        # ── Check Dynamic Lane Defense Adaptation (Rule 11) ────────────
        threat_lane = active_threat_lane
        if not threat_lane:
            for e in enemies:
                ey = (e.get('box', (0, 0, 0, 0))[1] + e.get('box', (0, 0, 0, 0))[3]) / 2.0 / cur_h
                if ey >= 0.36:
                    ex = (e.get('box', (0, 0, 0, 0))[0] + e.get('box', (0, 0, 0, 0))[2]) / 2.0 / cur_w
                    threat_lane = 'left' if ex < 0.50 else 'right'
                    break

        if threat_lane and card_name not in ('fireball', 'arrows'):
            play_lane = 'left' if pos_pct[0] < 0.50 else 'right'
            if play_lane != threat_lane and 0.45 <= pos_pct[1] <= 0.70:
                adjusted_x = int((1.0 - pos_pct[0]) * cur_w)
                print(f"🛡️ [RULE OVERRIDE] Adjusted deployment from {play_lane} to threatened {threat_lane} lane.")
                return {'action': 'play_card', 'card_slot': slot, 'position': (adjusted_x, pos[1]), 'tactical_rule': 'LANE_ADAPTATION'}

        return action

    # ── Master Arbiter ────────────────────────────────────────────────
    def arbitrate_decision(self, game_state, dt_action=None):
        mandatory = self.get_mandatory_action(game_state)
        if mandatory is not None:
            return mandatory

        if dt_action is not None:
            validated = self.validate_candidate_action(dt_action, game_state)
            if validated is not None:
                return validated

        # Elixir Relief Valve (Prevents Analysis Paralysis passivity trap):
        if game_state.get('elixir', 0.0) >= 9.0:
            leak_action = self.check_elixir_leak_prevention(game_state, min_elixir=9.0)
            if leak_action:
                return leak_action

        return None
