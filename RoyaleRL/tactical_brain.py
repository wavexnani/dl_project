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

import re
import time
import numpy as np
import config
from config import CARD_COSTS

# --- Complete Universal Threat Hierarchy (Covering all 133 Clash Royale units) ---
THREAT_LEVELS = {
    # ── Tier 1: Critical Win Conditions & Tower Demolishers (Threat 10) ──
    'giant': 10, 'hog': 10, 'balloon': 10, 'ramrider': 10, 'wallbreakers': 10,
    'skeletonbarrel': 10, 'goblinbarrel': 10, 'egiant': 10, 'golem': 10,
    'royalgiant': 10, 'battleram': 10, 'lavahound': 10, 'pig': 10, 'royalhogs': 10,
    'goblingiant': 10, 'evoroyalgiant': 10,

    # ── Tier 2: Heavy Bruisers & Threat Tanks (Threat 9) ──
    'mini-pekka': 9, 'pekka': 9, 'megaknight': 9, 'prince': 9, 'darkprince': 9,
    'ebarbs': 9, 'giantskeleton': 9, 'sparky': 9, 'goldenknight': 9, 'monk': 9,
    'skellyking': 9, 'evoknight': 9,

    # ── Tier 3: Champions, Tower Infiltrators & High Ranged DPS (Threat 8) ──
    'queen': 8, 'miner': 8, 'mightyminer': 8, 'graveyard': 8, 'goblindrill': 8,
    'musketeer': 8, 'wizard': 8, 'witch': 8, 'marcher': 8, 'executioner': 8,
    'dartgoblin': 8, 'flyingmachine': 8, 'littleprince': 8, 'hunter': 8,
    'motherwitch': 8, 'princess': 8, 'minions': 8, 'minionhorde': 8,
    'infernodragon': 8, 'babydragon': 8, 'edragon': 8,

    # ── Tier 4: Swarms & Fast Attackers (Threat 7) ──
    'goblin': 7, 'speargoblin': 7, 'skeleton': 7, 'guards': 7, 'bat': 7,
    'firecracker': 7, 'rascalboy': 7, 'rascalgirl': 7, 'barbarian': 7,
    'royalrecruits': 7, 'zappies': 7, 'evofirecracker': 7, 'evoskeleton': 7,
    'evobomber': 7, 'evobat': 7,

    # ── Tier 5: Mini-Tanks & Defensive Ground (Threat 5) ──
    'valk': 5, 'knight': 5, 'icegolem': 5, 'fisherman': 5, 'bowler': 5,
    'bandit': 5, 'royalghost': 5, 'battlehealer': 5, 'lumberjack': 5,
    'cannoncart': 5,

    # ── Tier 6: Spawners & Buildings (Threat 4) ──
    'goblin_hut': 4, 'barbarianhut': 4, 'furnace': 4, 'tombstone': 4,
    'goblin_cage': 3, 'archers': 3, 'evoarchers': 3, 'bomber': 3,
    'xbow': 8, 'mortar': 6
}

# --- Tactical Counter Matrix (Universal hard-counters for all 133 units) ---
COUNTER_MATRIX = {
    # ── Win Conditions ──
    'giant': ['mini-pekka', 'goblin_cage', 'musketeer', 'minions', 'knight'],
    'hog': ['mini-pekka', 'goblin_cage', 'minions', 'knight', 'musketeer'],
    'balloon': ['musketeer', 'archers', 'minions', 'arrows', 'fireball'],
    'ramrider': ['mini-pekka', 'goblin_cage', 'minions', 'knight'],
    'wallbreakers': ['arrows', 'minions', 'archers', 'knight'],
    'skeletonbarrel': ['arrows', 'musketeer', 'minions', 'archers'],
    'goblinbarrel': ['arrows', 'archers', 'knight', 'minions', 'valk'],
    'egiant': ['mini-pekka', 'goblin_cage', 'musketeer'],
    'golem': ['mini-pekka', 'minions', 'goblin_cage', 'musketeer'],
    'royalgiant': ['mini-pekka', 'minions', 'goblin_cage', 'musketeer'],
    'battleram': ['mini-pekka', 'goblin_cage', 'knight', 'minions'],
    'lavahound': ['musketeer', 'minions', 'archers'],
    'pig': ['arrows', 'fireball', 'minions', 'valk', 'knight', 'mini-pekka'],
    'royalhogs': ['arrows', 'fireball', 'valk', 'mini-pekka', 'knight', 'minions'],
    'goblingiant': ['mini-pekka', 'goblin_cage', 'minions', 'musketeer'],
    'evoroyalgiant': ['mini-pekka', 'minions', 'goblin_cage', 'musketeer'],

    # ── Heavy Bruisers ──
    'mini-pekka': ['minions', 'goblin_cage', 'knight', 'archers'],
    'pekka': ['minions', 'goblin_cage', 'knight', 'archers'],
    'megaknight': ['minions', 'mini-pekka', 'knight', 'goblin_cage'],
    'prince': ['goblin_cage', 'minions', 'knight', 'mini-pekka'],
    'darkprince': ['minions', 'mini-pekka', 'knight', 'goblin_cage'],
    'ebarbs': ['minions', 'goblin_cage', 'knight', 'valk', 'fireball'],
    'giantskeleton': ['minions', 'knight', 'goblin_cage', 'mini-pekka'],
    'sparky': ['minions', 'mini-pekka', 'knight', 'fireball'],
    'goldenknight': ['knight', 'mini-pekka', 'minions', 'goblin_cage'],
    'monk': ['minions', 'musketeer', 'mini-pekka', 'goblin_cage'],
    'skellyking': ['mini-pekka', 'knight', 'arrows', 'minions'],
    'evoknight': ['mini-pekka', 'minions', 'goblin_cage', 'musketeer'],

    # ── Champions & Ranged ──
    'queen': ['knight', 'mini-pekka', 'fireball', 'minions', 'musketeer'],
    'miner': ['knight', 'mini-pekka', 'archers', 'minions'],
    'mightyminer': ['minions', 'mini-pekka', 'knight', 'goblin_cage'],
    'graveyard': ['archers', 'minions', 'knight', 'arrows'],
    'goblindrill': ['knight', 'mini-pekka', 'valk', 'arrows'],
    'musketeer': ['knight', 'mini-pekka', 'fireball', 'minions'],
    'wizard': ['knight', 'mini-pekka', 'fireball'],
    'witch': ['valk', 'knight', 'fireball', 'mini-pekka', 'arrows'],
    'marcher': ['knight', 'mini-pekka', 'fireball', 'arrows'],
    'executioner': ['mini-pekka', 'knight', 'fireball'],
    'dartgoblin': ['arrows', 'knight', 'archers'],
    'flyingmachine': ['musketeer', 'fireball', 'archers', 'minions'],
    'littleprince': ['knight', 'mini-pekka', 'fireball', 'arrows'],
    'princess': ['arrows', 'knight', 'minions'],
    'hunter': ['knight', 'mini-pekka', 'fireball'],
    'motherwitch': ['fireball', 'knight', 'mini-pekka', 'arrows'],

    # ── Swarms ──
    'minions': ['arrows', 'musketeer', 'archers', 'fireball'],
    'minionhorde': ['arrows', 'fireball', 'musketeer', 'archers'],
    'goblin': ['arrows', 'archers', 'valk', 'knight'],
    'speargoblin': ['arrows', 'archers', 'knight', 'musketeer'],
    'skeleton': ['arrows', 'archers', 'knight'],
    'guards': ['arrows', 'valk', 'knight', 'archers'],
    'bat': ['arrows', 'musketeer', 'archers', 'minions'],
    'firecracker': ['arrows', 'knight', 'fireball', 'mini-pekka'],
    'evofirecracker': ['fireball', 'arrows', 'knight', 'mini-pekka'],
    'barbarian': ['fireball', 'valk', 'minions', 'arrows'],
    'royalrecruits': ['valk', 'fireball', 'minions', 'arrows'],
    'zappies': ['fireball', 'valk', 'knight'],

    # ── Mini-Tanks ──
    'valk': ['minions', 'mini-pekka', 'musketeer', 'goblin_cage'],
    'knight': ['mini-pekka', 'minions', 'musketeer', 'goblin_cage'],
    'icegolem': ['mini-pekka', 'musketeer', 'minions'],
    'bowler': ['minions', 'mini-pekka', 'musketeer'],
    'babydragon': ['musketeer', 'archers', 'minions'],
    'infernodragon': ['musketeer', 'archers', 'minions'],
    'edragon': ['musketeer', 'mini-pekka', 'archers'],
    'bandit': ['knight', 'mini-pekka', 'goblin_cage'],
    'royalghost': ['knight', 'mini-pekka', 'minions'],
    'lumberjack': ['mini-pekka', 'minions', 'knight', 'goblin_cage'],
    'fisherman': ['minions', 'mini-pekka', 'knight'],
    'cannoncart': ['mini-pekka', 'minions', 'knight'],

    # ── Spawners & Buildings ──
    'goblin_hut': ['fireball', 'musketeer', 'giant'],
    'goblin_cage': ['musketeer', 'minions', 'giant'],
    'barbarianhut': ['fireball', 'giant', 'musketeer'],
    'furnace': ['fireball', 'musketeer', 'giant'],
    'tombstone': ['arrows', 'musketeer', 'giant'],
    'xbow': ['giant', 'knight', 'mini-pekka', 'minions'],
    'mortar': ['knight', 'giant', 'musketeer', 'minions'],
    'archers': ['arrows', 'knight', 'fireball'],
    'evoarchers': ['arrows', 'knight', 'fireball']
}

# Flying units (STRICT ANTI-AIR LAW: Ground-only melee can NEVER target these!)
FLYING_UNITS = {
    'minions', 'minionhorde', 'balloon', 'babydragon', 'edragon',
    'infernodragon', 'skeletondragon', 'bat', 'lavahound',
    'flyingmachine', 'pheonix', 'skeletonbarrel', 'evobat'
}

# Cards that CANNOT hit air targets (STRICTLY FORBIDDEN against Minions and flying units)
GROUND_ONLY_MELEE = {'knight', 'mini-pekka'}

# Damage thresholds for spell execution (estimated tower damage)
SPELL_TOWER_DAMAGE = {
    'fireball': 280,     # Princess Tower damage
    'arrows': 140
}

# King Tower Zone in Normalized Coordinates (X: [0.36, 0.64], Y: [0.00, 0.22])
KING_TOWER_ZONE = (0.36, 0.00, 0.64, 0.22)


def normalize_card_name(name):
    """Normalizes card names across all 133 detector labels, OCR, and deck lists."""
    if not name:
        return ''
    n = str(name).lower().strip().replace('_', '').replace('-', '').replace(' ', '')
    n_dedup = re.sub(r'([a-z])\1+', r'\1', n)

    def match(pattern):
        return pattern in n or pattern in n_dedup

    if match('minipekka'):
        return 'mini-pekka'
    if match('megaknight'):
        return 'megaknight'
    if match('pekka'):
        return 'pekka'
    if match('queen'):
        return 'queen'
    if match('mightyminer'):
        return 'mightyminer'
    if match('miner'):
        return 'miner'
    if match('valk'):
        return 'valk'
    if match('spear'):
        return 'speargoblin'
    if match('goblinbarrel'):
        return 'goblinbarrel'
    if match('goblincage'):
        return 'goblin_cage'
    if match('goblinhut'):
        return 'goblin_hut'
    if match('goblindrill'):
        return 'goblindrill'
    if match('goblingiant'):
        return 'goblingiant'
    if match('goblin'):
        return 'goblin'
    if match('hog') or match('pig'):
        return 'hog'
    if match('balloon'):
        return 'balloon'
    if match('minion'):
        return 'minions'
    if match('archer') and not match('magic') and not match('queen'):
        return 'archers'
    if match('giant') and not match('skel'):
        return 'giant'
    if match('knight') and not match('golden') and not match('mega'):
        return 'knight'
    if match('musk'):
        return 'musketeer'
    if match('fireball'):
        return 'fireball'
    if match('arrow'):
        return 'arrows'
    if match('skel'):
        if match('barrel'):
            return 'skeletonbarrel'
        if match('giant'):
            return 'giantskeleton'
        if match('king'):
            return 'skellyking'
        return 'skeleton'
    if match('rascal'):
        if match('girl'):
            return 'rascalgirl'
        return 'rascalboy'
    if match('firecracker'):
        return 'firecracker'
    if match('wizard'):
        return 'wizard'
    if match('witch'):
        return 'witch'
    if match('prince') and not match('dark'):
        return 'prince'
    if match('darkprince'):
        return 'darkprince'
    if match('ebarbs'):
        return 'ebarbs'
    if match('sparky'):
        return 'sparky'
    if match('wallbreaker'):
        return 'wallbreakers'
    if match('ramrider'):
        return 'ramrider'
    if match('battleram'):
        return 'battleram'
    return n


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
        """Safely parses HP value from OCR data, stripping OCR level-badge noise."""
        if hp_val is None:
            return None
        try:
            cleaned = str(hp_val).strip().replace(',', '').replace(' ', '')
            val = int(cleaned)
            # If val > 5000, it's often a leading level badge digit (e.g. 41624 -> 1624)
            if val > 5000:
                s = str(val)
                while len(s) > 4:
                    s = s[1:]
                val = int(s)
            return val if 0 < val <= 5000 else None
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
            if hp is not None and hp >= 40:
                if is_king:
                    ptl_hp = self._parse_hp(ocr_data.get('ptl'))
                    ptr_hp = self._parse_hp(ocr_data.get('ptr'))
                    # Strictly enforce King Tower dormancy: NEVER snipe King if BOTH Princess Towers are alive!
                    if ptl_hp is not None and ptr_hp is not None:
                        continue
                    if hp > SPELL_TOWER_DAMAGE['fireball']:
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
            if center_y >= 0.26:
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

        # Only trigger if realistic damage drop (30 to 1200) - rejects OCR noise jumps
        if self.last_pbl_hp is not None and pbl_hp is not None and 30 <= (self.last_pbl_hp - pbl_hp) <= 1200:
            print(f"🚨 [TOWER SENSOR] Left Tower damaged ({self.last_pbl_hp} -> {pbl_hp})! Locking emergency defense on LEFT lane!")
            lane_enemies = [e for e in enemies if ((e.get('box', (0, 0, 0, 0))[0] + e.get('box', (0, 0, 0, 0))[2]) / 2.0 / cur_w) < 0.50]
            threat_name = normalize_card_name(lane_enemies[0].get('name')) if lane_enemies else 'giant'
            threat_score = THREAT_LEVELS.get(threat_name, 10)
            self.active_threat_memory.append({
                'name': threat_name,
                'score': threat_score,
                'lane': 'left',
                'x': 0.25,
                'y': 0.60,
                'box': (int(0.20 * cur_w), int(0.55 * cur_h), int(0.30 * cur_w), int(0.65 * cur_h)),
                'timestamp': now
            })
        elif self.last_pbr_hp is not None and pbr_hp is not None and 30 <= (self.last_pbr_hp - pbr_hp) <= 1200:
            print(f"🚨 [TOWER SENSOR] Right Tower damaged ({self.last_pbr_hp} -> {pbr_hp})! Locking emergency defense on RIGHT lane!")
            lane_enemies = [e for e in enemies if ((e.get('box', (0, 0, 0, 0))[0] + e.get('box', (0, 0, 0, 0))[2]) / 2.0 / cur_w) >= 0.50]
            threat_name = normalize_card_name(lane_enemies[0].get('name')) if lane_enemies else 'giant'
            threat_score = THREAT_LEVELS.get(threat_name, 10)
            self.active_threat_memory.append({
                'name': threat_name,
                'score': threat_score,
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

        # Sort threats by severity + proximity to our towers
        self.active_threat_memory.sort(key=lambda t: (t['score'] * 10.0 + t['y'] * 15.0), reverse=True)
        top_threat = self.active_threat_memory[0]
        t_name = top_threat['name']
        threat_lane = 'left' if top_threat['x'] < 0.50 else 'right'
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
                if t_name in FLYING_UNITS and c in GROUND_ONLY_MELEE:
                    continue
                # For Tanks / Win Conditions, NEVER pick spells as body-blocking pull cards!
                if t_name in ('giant', 'golem', 'pekka', 'megaknight', 'hog', 'prince', 'ramrider', 'battleram', 'balloon') and c in ('arrows', 'fireball'):
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
        # A. Swarms (Minions, Bats, Goblins, etc.): Direct Arrows strike with predictive lead aiming
        if t_name in ('minions', 'minionhorde', 'bat', 'speargoblin', 'goblin', 'skeleton', 'firecracker'):
            if best_counter == 'arrows':
                strike_pos = self.calculate_predictive_aim(t_box, enemy_name=t_name, spell_name='arrows')
                print(f"🏹 [ANTI-AIR CLEAR] Casting ARROWS with predictive lead aiming at {strike_pos}!")
                return {'action': 'play_card', 'card_slot': counter_slot, 'position': strike_pos, 'tactical_rule': 'AIR_SWARM_CLEAR'}
            elif t_name in FLYING_UNITS:
                plant_x_pct = 0.23 if threat_lane == 'left' else 0.76
                plant_y_pct = 0.72
                deploy_pos = self._to_pixels(plant_x_pct, plant_y_pct)
                print(f"🛡️ [ANTI-AIR DEFENSE] Deploying {best_counter.upper()} behind tower to intercept {t_name.upper()}!")
                return {'action': 'play_card', 'card_slot': counter_slot, 'position': deploy_pos, 'tactical_rule': 'RANGED_ANTI_AIR'}

        # B. Tower Infiltrators (Miner, Goblin Barrel directly on tower): Intercept directly on tower
        if t_name in ('miner', 'mightyminer', 'goblinbarrel') or (top_threat['y'] >= 0.58 and t_name in ('valk', 'goblin', 'guards')):
            t_px_x = int((t_box[0] + t_box[2]) / 2)
            t_px_y = int((t_box[1] + t_box[3]) / 2)
            print(f"🛡️ [TOWER INTERCEPT] Deploying {best_counter.upper()} directly onto {t_name.upper()} at ({t_px_x}, {t_px_y})!")
            return {'action': 'play_card', 'card_slot': counter_slot, 'position': (t_px_x, t_px_y), 'tactical_rule': 'TOWER_INTERCEPT'}

        # C. Single-Target Melee / Tanks / Win Conditions: Center-Pull Kiting vs Direct Intercept
        if t_name in ('giant', 'mini-pekka', 'knight', 'hog', 'balloon', 'pekka', 'megaknight', 'prince', 'darkprince', 'ebarbs', 'ramrider', 'golem', 'egiant', 'royalgiant'):
            if top_threat['y'] >= 0.65:
                # Danger zone! Threat is directly assaulting friendly Princess Tower -> drop counter right on top!
                t_px_x = int((t_box[0] + t_box[2]) / 2)
                t_px_y = int((t_box[1] + t_box[3]) / 2)
                print(f"🛡️ [CLOSE DEFENSE] Deploying {best_counter.upper()} directly onto {t_name.upper()} at ({t_px_x}, {t_px_y})!")
                return {'action': 'play_card', 'card_slot': counter_slot, 'position': (t_px_x, t_px_y), 'tactical_rule': 'TOWER_INTERCEPT'}
            else:
                plant_x_pct = 0.47 if threat_lane == 'left' else 0.53
                plant_y_pct = 0.63
                deploy_pos = self._to_pixels(plant_x_pct, plant_y_pct)
                print(f"🛡️ [CENTER PULL] Playing {best_counter.upper()} at Center Kiting Zone ({plant_x_pct:.2f}, {plant_y_pct:.2f}) vs {t_name.upper()}!")
                return {'action': 'play_card', 'card_slot': counter_slot, 'position': deploy_pos, 'tactical_rule': 'CENTER_PULL_DEFENSE'}

        # D. Ranged Attackers & Champions (Musketeer, Queen, Wizard, Witch, etc.): Drop melee directly on top
        if t_name in ('musketeer', 'queen', 'wizard', 'witch', 'marcher', 'dartgoblin', 'executioner', 'littleprince', 'princess') and best_counter in ('knight', 'mini-pekka'):
            drop_x = int((t_box[0] + t_box[2]) / 2)
            drop_y = int((t_box[1] + t_box[3]) / 2)
            print(f"⚔️ [MELEE DROP] Dropping {best_counter.upper()} directly on {t_name.upper()} at ({drop_x}, {drop_y})!")
            return {'action': 'play_card', 'card_slot': counter_slot, 'position': (drop_x, drop_y), 'tactical_rule': 'MELEE_ON_RANGED'}

        # E. Standard Lane Defense
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
                    ptl_hp = self._parse_hp(ocr_data.get('ptl'))
                    ptr_hp = self._parse_hp(ocr_data.get('ptr'))
                    left_dead = (ptl_hp is None)
                    right_dead = (ptr_hp is None)

                    # Strategic Lane Route:
                    # If one tower is already dead, push reinforcements down the BREACHED lane
                    # directly towards the King Tower, rather than restarting on the second tower!
                    if left_dead and not right_dead:
                        cycle_lane = 'left'
                    elif right_dead and not left_dead:
                        cycle_lane = 'right'
                    else:
                        p_left = ptl_hp if ptl_hp is not None else 2534
                        p_right = ptr_hp if ptr_hp is not None else 2534
                        cycle_lane = 'left' if p_left <= p_right else 'right'

                    x_pct = 0.26 if cycle_lane == 'left' else 0.74
                    y_pct = 0.76  # Safely behind King Tower on arena grass (well above card UI)
                    deploy_pos = self._to_pixels(x_pct, y_pct)
                    print(f"⚡ [LEAK PREVENTION] Elixir at {elixir:.1f}! Cycling {c.upper()} safely behind King Tower ({x_pct:.2f}, {y_pct:.2f})!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': deploy_pos, 'tactical_rule': 'ELIXIR_LEAK_CYCLE'}

        return None

    # ── Rule 10: The Breach & Pocket King Assault ─────────────────────
    def check_pocket_deployment(self, game_state):
        """
        Rule 10: Once one enemy Princess Tower is destroyed, the AI exploits the breach
        to spearhead deep into enemy territory and assault the King Tower for a decisive 3-crown win,
        while maintaining sentinel defense on the other lane.
        """
        ocr_data = game_state.get('ocr_data', {})
        hand = [normalize_card_name(c) for c in game_state.get('hand', [])]
        elixir = game_state.get('elixir', 0.0)

        left_dead = (self._parse_hp(ocr_data.get('ptl')) is None)
        right_dead = (self._parse_hp(ocr_data.get('ptr')) is None)

        pocket_units = ['giant', 'mini-pekka', 'musketeer', 'knight', 'archers']

        # Case 1: Left Tower is down -> Push through Left Breach into King Tower!
        if left_dead and not right_dead:
            for u in pocket_units:
                if u in hand and elixir >= CARD_COSTS.get(u, 4):
                    slot = hand.index(u)
                    pos = self._to_pixels(0.42, 0.42)
                    print(f"🔥 [BREACH SPEARHEAD] Left tower down! Deploying {u.upper()} through breach to assault KING TOWER!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': pos, 'tactical_rule': 'POCKET_ASSAULT'}

        # Case 2: Right Tower is down -> Push through Right Breach into King Tower!
        if right_dead and not left_dead:
            for u in pocket_units:
                if u in hand and elixir >= CARD_COSTS.get(u, 4):
                    slot = hand.index(u)
                    pos = self._to_pixels(0.58, 0.42)
                    print(f"🔥 [BREACH SPEARHEAD] Right tower down! Deploying {u.upper()} through breach to assault KING TOWER!")
                    return {'action': 'play_card', 'card_slot': slot, 'position': pos, 'tactical_rule': 'POCKET_ASSAULT'}

        # Case 3: Both Towers down -> Direct Center Breach on King Tower!
        if left_dead and right_dead:
            for u in pocket_units:
                if u in hand and elixir >= CARD_COSTS.get(u, 4):
                    slot = hand.index(u)
                    pos = self._to_pixels(0.50, 0.40)
                    print(f"🔥 [DOUBLE BREACH] Both towers down! Deploying {u.upper()} center for instant 3-Crown!")
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
            if center_y >= 0.26:
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
                normalize_card_name(e.get('name')) in FLYING_UNITS
                for e in enemies
                if (e.get('box', (0, 0, 0, 0))[1] + e.get('box', (0, 0, 0, 0))[3]) / 2.0 / cur_h >= 0.26
            ) or any(t['name'] in FLYING_UNITS for t in self.active_threat_memory)
            if has_air_threat:
                print(f"⛔ [RULE OVERRIDE] Blocked {card_name.upper()} deployment against Air Threat (Ground melee cannot hit air!).")
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

        # ── Check Spell Spatial Value & Waste Rules (Rule 8) ────────────
        if card_name in ('arrows', 'fireball'):
            enemy_positions = []
            for e in enemies:
                bx = e.get('box', (0, 0, 0, 0))
                ex_pct = ((bx[0] + bx[2]) / 2.0) / cur_w
                ey_pct = ((bx[1] + bx[3]) / 2.0) / cur_h
                enemy_positions.append((ex_pct, ey_pct))

            spell_radius = 0.14 if card_name == 'arrows' else 0.10
            has_enemy_in_radius = any(
                np.hypot(pos_pct[0] - ex, pos_pct[1] - ey) <= (spell_radius + 0.05)
                for ex, ey in enemy_positions
            )
            is_targeting_tower = (pos_pct[1] <= 0.22)

            # Friendly base / Friendly towers (pos_pct[1] >= 0.45):
            if pos_pct[1] >= 0.45:
                # STRICT: NEVER throw spells on friendly ground unless enemy is inside splash radius!
                if not has_enemy_in_radius:
                    print(f"⛔ [RULE OVERRIDE] Blocked wasting {card_name.upper()} on friendly base/towers with no enemies in splash radius.")
                    return None

            # Enemy territory (pos_pct[1] < 0.45):
            else:
                if not is_targeting_tower and not has_enemy_in_radius:
                    print(f"⛔ [RULE OVERRIDE] Blocked wasting {card_name.upper()} on empty enemy grass with no tower or troops.")
                    return None

        # ── Check Fragile Troop Protection (Rule 9) ────────────────────
        if card_name in ('musketeer', 'archers') and pos_pct[1] < 0.52:
            safe_pos = (pos[0], int(pos[1] + 0.14 * cur_h))
            print(f"🛡️ [RULE OVERRIDE] Pulled {card_name.upper()} back behind river for safety.")
            return {'action': 'play_card', 'card_slot': slot, 'position': safe_pos, 'tactical_rule': 'TANK_IN_FRONT_SEQUENCING'}

        # ── Check Valid Troop Deployment Bounds (Rule 20) ──────────────
        # Troops (non-spells) cannot be placed on enemy grass (Y < 0.50) unless that tower is destroyed
        if card_name not in ('fireball', 'arrows') and pos_pct[1] < 0.50:
            ocr_data = game_state.get('ocr_data', {})
            left_dead = (self._parse_hp(ocr_data.get('ptl')) is None)
            right_dead = (self._parse_hp(ocr_data.get('ptr')) is None)
            target_lane = 'left' if pos_pct[0] < 0.50 else 'right'
            tower_destroyed = (left_dead if target_lane == 'left' else right_dead)

            if not tower_destroyed:
                safe_y = int(0.54 * cur_h)
                clamped_pos = (pos[0], safe_y)
                print(f"🛡️ [RULE OVERRIDE] Clamped illegal deployment of {card_name.upper()} back to friendly side ({pos_pct[0]:.2f}, 0.54).")
                return {'action': 'play_card', 'card_slot': slot, 'position': clamped_pos, 'tactical_rule': 'VALID_DEPLOYMENT_CLAMP'}

        # ── Check Dynamic Lane Defense Adaptation (Rule 11) ────────────
        threat_lane = active_threat_lane
        if not threat_lane:
            for e in enemies:
                ey = (e.get('box', (0, 0, 0, 0))[1] + e.get('box', (0, 0, 0, 0))[3]) / 2.0 / cur_h
                if ey >= 0.26:
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

    def has_unresolved_threat(self, game_state):
        """
        Returns True if an active high-threat enemy (score >= 7) is on our side (y >= 0.26)
        and we must hold elixir for the counter rather than allowing the AI to waste it.
        """
        now = time.time()
        cur_w, cur_h = self._get_res()
        enemies = game_state.get('enemies', [])
        for e in enemies:
            name = normalize_card_name(e.get('name', ''))
            box = e.get('box', (0, 0, 0, 0))
            cy = (box[1] + box[3]) / 2.0 / cur_h
            if cy >= 0.26 and THREAT_LEVELS.get(name, 0) >= 7:
                return True
        valid_memory = [t for t in self.active_threat_memory if (now - t['timestamp']) < 4.0 and t.get('score', 0) >= 7 and t.get('y', 0) >= 0.26]
        return len(valid_memory) > 0

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
