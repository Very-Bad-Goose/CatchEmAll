import json
import numpy as np
import cv2
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from collections import deque
import time
import os

class MGBAFireRedEnv(Env):
    """Pokemon FireRed environment using mGBA via file communication."""

    # Resolve project root based on this file's location
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    IPC_DIR = os.path.join(PROJECT_ROOT, "ipc")

    # Ensure IPC directory exists
    os.makedirs(IPC_DIR, exist_ok=True)

    def __init__(self):
        super().__init__()

        # IPC file paths (relative to project root)
        self.state_file = os.path.join(self.IPC_DIR, "mgba_state.txt")
        self.action_file = os.path.join(self.IPC_DIR, "mgba_action.txt")

        # Observation space (stacked frames)
        self.observation_space = Box(
            low=0, high=1, shape=(4, 84, 84), dtype=np.float32
        )

        # Action space
        self.action_space = Discrete(9)
        self.actions = ["NONE", "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]

        # Frame stack
        self.frame_stack = deque(maxlen=4)

        # Tracking
        self.prev_state = None
        self.step_count = 0

        print("=" * 60)
        print("mGBA FireRed Environment (File-based)")
        print("=" * 60)
        print("Waiting for mGBA to start...")
        print("1. Open mGBA")
        print("2. Load Pokemon FireRed ROM")
        print("3. Tools → Scripting → File → Load pokemon_firered.lua")
        print("=" * 60)

        # Wait for first state file
        self._wait_for_mgba()

    
    def _wait_for_mgba(self, timeout=30):
        """Wait for mGBA to create state file."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists(self.state_file):
                print("✓ mGBA detected!")
                time.sleep(0.5)  # Let it stabilize
                return
            time.sleep(0.5)
            if int(time.time() - start_time) % 5 == 0:
                print(f"Still waiting... ({int(time.time() - start_time)}s)")
        
        raise Exception(f"mGBA not detected after {timeout}s. Is the Lua script loaded?")
    
    def _read_state(self):
        """Read game state from file."""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if not os.path.exists(self.state_file):
                    time.sleep(0.01)
                    continue
                
                with open(self.state_file, 'r') as f:
                    data = f.read().strip()
                
                if not data:
                    time.sleep(0.01)
                    continue
                
                state = json.loads(data)
                return state
            
            except (json.JSONDecodeError, FileNotFoundError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.01)
                    continue
                else:
                    # Return previous state if available
                    if self.prev_state:
                        return self.prev_state
                    # Otherwise return safe defaults
                    return {
                        'in_battle': 0,
                        'player_hp': 100,
                        'player_hp_max': 100,
                        'opp_hp': 0,
                        'opp_hp_max': 1,
                        'map_group': 0,
                        'map_num': 0,
                        'badges': 0
                    }
        
        return self.prev_state or {
            'in_battle': 0, 'player_hp': 100, 'player_hp_max': 100,
            'opp_hp': 0, 'opp_hp_max': 1, 'map_group': 0, 'map_num': 0, 'badges': 0
        }
    
    def _write_action(self, action):
        """Write action to file for mGBA."""
        action_str = self.actions[action]
        try:
            with open(self.action_file, 'w') as f:
                f.write(action_str)
        except Exception as e:
            print(f"Error writing action: {e}")
    
    def _create_frame_from_state(self, state):
        """Create a visual frame from game state."""
        # Create base frame
        frame = np.zeros((240, 160, 3), dtype=np.uint8)
        
        # Background color based on map
        map_seed = state['map_group'] * 100 + state['map_num']
        np.random.seed(map_seed)
        bg_color = np.random.randint(30, 100, 3)
        frame[:] = bg_color
        
        # HP bar (green)
        if state['player_hp_max'] > 0:
            hp_ratio = min(state['player_hp'] / state['player_hp_max'], 1.0)
            bar_width = int(140 * hp_ratio)
            cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 10), (150, 30), (255, 255, 255), 2)
        
        # Battle indicator (red box)
        if state['in_battle'] > 0:
            cv2.rectangle(frame, (50, 90), (110, 150), (255, 0, 0), -1)
            cv2.putText(frame, "BATTLE", (55, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Opponent HP
            if state['opp_hp_max'] > 0:
                opp_hp_ratio = min(state['opp_hp'] / state['opp_hp_max'], 1.0)
                opp_bar_width = int(140 * opp_hp_ratio)
                cv2.rectangle(frame, (10, 50), (10 + opp_bar_width, 70), (255, 0, 0), -1)
                cv2.rectangle(frame, (10, 50), (150, 70), (255, 255, 255), 2)
        
        # Badge count
        cv2.putText(frame, f"Badges: {state['badges']}", (10, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def preprocess_frame(self, frame):
        """Preprocess frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return normalized
    
    def stack_frames(self, frame):
        """Stack frames."""
        if len(self.frame_stack) == 0:
            for _ in range(4):
                self.frame_stack.append(frame)
        else:
            self.frame_stack.append(frame)
        return np.array(self.frame_stack, dtype=np.float32)
    
    def calculate_reward(self, state):
        """Calculate reward from game state."""
        if not self.prev_state:
            return 0.0
        
        reward = 0.0
        
        # Badge reward (HUGE)
        if state['badges'] > self.prev_state.get('badges', 0):
            reward += 1000
            print(f"🎖️  BADGE EARNED! Total: {state['badges']}/8")
        
        # Battle rewards
        if state['in_battle'] > 0 and state['opp_hp_max'] > 0:
            # Damage dealt
            prev_opp_hp = self.prev_state.get('opp_hp', state['opp_hp_max'])
            opp_damage = prev_opp_hp - state['opp_hp']
            if opp_damage > 0:
                reward += opp_damage * 0.5
            
            # Defeated opponent
            if prev_opp_hp > 0 and state['opp_hp'] == 0:
                reward += 100
                print("💥 Opponent defeated!")
        
        # Damage taken penalty
        if state['player_hp_max'] > 0:
            prev_player_hp = self.prev_state.get('player_hp', state['player_hp_max'])
            player_damage = prev_player_hp - state['player_hp']
            if player_damage > 0:
                reward -= player_damage * 0.3
            
            # Fainted
            if state['player_hp'] == 0 and prev_player_hp > 0:
                reward -= 50
                print("💀 Pokemon fainted!")
        
        # Map change (exploration)
        prev_map = (self.prev_state.get('map_group', 0), self.prev_state.get('map_num', 0))
        curr_map = (state['map_group'], state['map_num'])
        if curr_map != prev_map and curr_map != (0, 0):
            reward += 5
            print(f"📍 New area: Group {curr_map[0]}, Map {curr_map[1]}")
        
        # Small time penalty
        reward -= 0.01
        
        return reward
    
    def step(self, action):
        """Execute one step."""
        # Write action
        self._write_action(action)
        
        # Wait a bit for mGBA to process
        time.sleep(0.016)  # ~60 FPS
        
        # Read new state
        state = self._read_state()
        
        # Create visual frame
        frame = self._create_frame_from_state(state)
        preprocessed = self.preprocess_frame(frame)
        stacked_obs = self.stack_frames(preprocessed)
        
        # Calculate reward
        reward = self.calculate_reward(state)
        
        # Update tracking
        self.prev_state = state
        self.step_count += 1
        
        # Episode termination
        done = self.step_count >= 10000
        
        info = {
            'in_battle': state['in_battle'],
            'player_hp': state['player_hp'],
            'player_hp_max': state['player_hp_max'],
            'opp_hp': state['opp_hp'],
            'badges': state['badges'],
            'map': (state['map_group'], state['map_num']),
            'step_count': self.step_count
        }
        
        return stacked_obs, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            super().reset(seed=seed)
        
        # Just clear our tracking (mGBA stays running)
        self.step_count = 0
        
        # Read initial state
        state = self._read_state()
        
        # Create initial frame
        frame = self._create_frame_from_state(state)
        
        # Clear frame stack
        self.frame_stack.clear()
        preprocessed = self.preprocess_frame(frame)
        stacked_obs = self.stack_frames(preprocessed)
        
        # Initialize tracking
        self.prev_state = state
        
        info = {
            'player_hp': state['player_hp'],
            'badges': state['badges'],
            'map': (state['map_group'], state['map_num'])
        }
        
        return stacked_obs, info
    
    def close(self):
        """Cleanup."""
        # Clean up files
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            if os.path.exists(self.action_file):
                os.remove(self.action_file)
        except:
            pass
        print("Environment closed")


# Test function
def test_env():
    """Test the environment."""
    print("\nTesting mGBA FireRed Environment...")
    print("-" * 60)
    
    try:
        env = MGBAFireRedEnv()
        
        print("\nResetting environment...")
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial HP: {info['player_hp']}")
        print(f"  Initial Badges: {info['badges']}")
        print(f"  Initial Map: {info['map']}")
        
        print("\nTaking 50 random actions...")
        for i in range(50):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            
            if i % 10 == 0 or reward != -0.01:  # Only print interesting steps
                print(f"Step {i+1}: Action={env.actions[action]:<6} "
                      f"Reward={reward:>7.2f} "
                      f"HP={info['player_hp']:>3}/{info['player_hp_max']:<3} "
                      f"Badges={info['badges']} "
                      f"Battle={'Yes' if info['in_battle'] else 'No'}")
            
            if done:
                print("Episode done!")
                break
        
        env.close()
        print("\n✓ Test completed successfully!")
        print("\nNow you can train with:")
        print("  python train_model.py --train --algorithm DQN")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_env()