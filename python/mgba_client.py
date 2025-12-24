import socket
import json
import numpy as np
import cv2
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from collections import deque
import time

class MGBAFireRedEnv(Env):
    """Pokemon FireRed environment using mGBA via socket."""
    
    def __init__(self, host='127.0.0.1', port=5005):
        super().__init__()
        
        self.host = host
        self.port = port
        self.socket = None
        
        # Observation space (stacked frames)
        self.observation_space = Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)
        
        # Action space
        self.action_space = Discrete(9)
        self.actions = ["NONE", "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
        
        # Frame stack
        self.frame_stack = deque(maxlen=4)
        
        # Tracking
        self.prev_state = None
        self.step_count = 0
        self.last_hp = 0
        self.last_map = (0, 0)
        
        # Connect to mGBA
        print("="*60)
        print("Connecting to mGBA...")
        print("Make sure mGBA is running with the Lua script loaded!")
        print("="*60)
        self._connect()
    
    def _connect(self):
        """Connect to mGBA."""
        max_attempts = 30
        for i in range(max_attempts):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(1.0)
                print(f"✓ Connected to mGBA on port {self.port}")
                return
            except ConnectionRefusedError:
                if i == 0:
                    print(f"Waiting for mGBA... (attempt {i+1}/{max_attempts})")
                time.sleep(1)
                if i == max_attempts - 1:
                    raise Exception("Could not connect to mGBA. Is it running with the Lua script?")
    
    def _receive_state(self):
        """Receive game state from mGBA."""
        try:
            data = self.socket.recv(4096).decode('utf-8').strip()
            if not data:
                raise ConnectionError("Connection lost")
            
            # Parse JSON state
            state = json.loads(data)
            return state
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Received data: {data}")
            return self.prev_state if self.prev_state else {
                'in_battle': 0, 'player_hp': 0, 'player_hp_max': 1,
                'opp_hp': 0, 'opp_hp_max': 1, 'map_group': 0, 'map_num': 0
            }
        except Exception as e:
            print(f"Error receiving state: {e}")
            return self.prev_state if self.prev_state else {
                'in_battle': 0, 'player_hp': 0, 'player_hp_max': 1,
                'opp_hp': 0, 'opp_hp_max': 1, 'map_group': 0, 'map_num': 0
            }
    
    def _send_action(self, action):
        """Send action to mGBA."""
        try:
            action_str = self.actions[action]
            self.socket.sendall(f"{action_str}\n".encode('utf-8'))
        except Exception as e:
            print(f"Error sending action: {e}")
            raise
    
    def _get_screen_from_mgba(self):
        """
        Get screenshot from mGBA.
        Note: This requires additional Lua script support for sending frames.
        For now, we'll use a placeholder based on game state.
        """
        # TODO: Implement frame capture via Lua
        # For now, return a simple state-based representation
        state = self.prev_state
        
        # Create a simple visualization based on state
        frame = np.zeros((240, 160, 3), dtype=np.uint8)
        
        # Different colors for different maps
        map_color = ((state['map_group'] * 30) % 255, (state['map_num'] * 50) % 255, 100)
        frame[:] = map_color
        
        # HP bar visualization
        if state['player_hp_max'] > 0:
            hp_ratio = state['player_hp'] / state['player_hp_max']
            bar_width = int(160 * hp_ratio)
            cv2.rectangle(frame, (0, 0), (bar_width, 20), (0, 255, 0), -1)
        
        # Battle indicator
        if state['in_battle'] > 0:
            cv2.rectangle(frame, (60, 100), (100, 140), (255, 0, 0), -1)
        
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
        
        # Damage dealt to opponent
        if state['in_battle'] > 0 and state['opp_hp_max'] > 0:
            prev_opp_hp = self.prev_state.get('opp_hp', state['opp_hp_max'])
            opp_damage = prev_opp_hp - state['opp_hp']
            if opp_damage > 0:
                reward += opp_damage / 10.0
        
        # Opponent defeated
        if self.prev_state.get('opp_hp', 1) > 0 and state['opp_hp'] == 0:
            reward += 50
            print("💥 Defeated opponent!")
        
        # Damage taken penalty
        if state['player_hp_max'] > 0:
            prev_player_hp = self.prev_state.get('player_hp', state['player_hp_max'])
            player_damage = prev_player_hp - state['player_hp']
            if player_damage > 0:
                reward -= player_damage / 5.0
        
        # Player fainted penalty
        if state['player_hp'] == 0 and self.prev_state.get('player_hp', 1) > 0:
            reward -= 100
            print("💀 Pokemon fainted!")
        
        # Map change reward (exploration)
        prev_map = (self.prev_state.get('map_group', 0), self.prev_state.get('map_num', 0))
        curr_map = (state['map_group'], state['map_num'])
        if curr_map != prev_map and curr_map != (0, 0):
            reward += 10
            print(f"📍 New area: {curr_map}")
        
        # Small time penalty
        reward -= 0.01
        
        return reward
    
    def step(self, action):
        """Execute one step."""
        # Send action
        self._send_action(action)
        
        # Receive new state
        state = self._receive_state()
        
        # Get frame (placeholder for now)
        frame = self._get_screen_from_mgba()
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
            'map': (state['map_group'], state['map_num']),
            'step_count': self.step_count
        }
        
        return stacked_obs, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            super().reset(seed=seed)
        
        # Send reset signal (or just clear state)
        self._send_action(0)  # Send NONE action
        
        # Receive initial state
        state = self._receive_state()
        
        # Get initial frame
        frame = self._get_screen_from_mgba()
        
        # Clear frame stack
        self.frame_stack.clear()
        preprocessed = self.preprocess_frame(frame)
        stacked_obs = self.stack_frames(preprocessed)
        
        # Reset tracking
        self.prev_state = state
        self.step_count = 0
        
        info = {
            'player_hp': state['player_hp'],
            'map': (state['map_group'], state['map_num'])
        }
        
        return stacked_obs, info
    
    def close(self):
        """Close connection."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            print("Connection closed")


# Test function
def test_env():
    """Test the environment."""
    print("\nTesting mGBA Environment...")
    print("-" * 60)
    
    try:
        env = MGBAFireRedEnv()
        
        print("\nResetting environment...")
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial HP: {info['player_hp']}")
        print(f"  Initial Map: {info['map']}")
        
        print("\nTaking 20 random actions...")
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            
            print(f"Step {i+1}: Action={env.actions[action]:<6} "
                  f"Reward={reward:>6.2f} "
                  f"HP={info['player_hp']:>3}/{info['player_hp_max']:<3} "
                  f"Battle={info['in_battle']>0}")
            
            if done:
                print("Episode done!")
                break
        
        env.close()
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_env()