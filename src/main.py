from fileutils import saveJson
from transformer import Transformer as AIModel
defaultData = {
    "batch_size": 16,
    "block_size": 32,
    "max_iters": 5000,
    "eval_interval": 100,
    "eval_iters": 200,
    "learning_rate": 1e-3,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.0,
    "input_file": 'resource\input.txt',
    "seed": 1337 #DEFAULT SEED
}

    
def main():
    # Get Input to change default data
    newData = defaultData
    # Make Edits
    saveJson(newData) # Save data to json for program to use
    AIModel()
if __name__ == "__main__":
    main()