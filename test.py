import argparse
from env import TextWorldEnvironment
from config import PPOConfig

def play_textworld(config, seed: int = None, difficulty: str = "easy"):
    # Initialize the environment with the provided seed or a random one
    env = TextWorldEnvironment(config=config)
    state = env.reset()
    print("*******Welcome to the TextWorld Game!*******")
    print(f"\nGame Seed: {env.seed}")
    print("\nInitial State:")
    print(state)
    print("\nType 'help' for commands, 'quit' to exit, 'restart' to start a new game.")

    while not env.done:
        # Display valid actions
        valid_actions = env.get_valid_actions()
        print("\nValid actions:", ", ".join(valid_actions))

        # Get user input
        action = input("\nYour action: ").strip().lower()

        if action == 'quit':
            print("*******Thanks for playing!*******")
            break
        elif action == 'help':
            print("\nCommands:")
            print("- Type any valid action (e.g., 'look', 'north', 'take key')")
            print("- Type 'help' to see this message")
            print("- Type 'quit' to exit the game")
            print("- Type 'restart' to start a new random game")
            continue
        elif action == 'restart':
            print("\nStarting a new random game...")
            state = env.reset()
            print(f"\nNew Game Seed: {env.seed}")
            print("\nNew State:")
            print(state)
            continue

        # Execute the action
        state, reward, done, info = env.step(action)

        # Display results
        print("\nState:")
        print(state)
        print(f"Reward: {reward:.2f}")
        print(f"Score: {info['score']}")
        if info.get('won', False):
            print("Congratulations! You won the game!")
        elif info.get('lost', False):
            print("Game over. You lost.")
        elif done:
            print("Game ended (max steps reached or other condition).")

    # Clean up
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a simple TextWorld game.")
    parser.add_argument('--seed', type=int, default=None, help='Random seed for game generation')
    parser.add_argument('--difficulty', type=str, default='medium', help='Difficulty level')
    args = parser.parse_args()
    config = PPOConfig()

    play_textworld(config, args.seed, args.difficulty)