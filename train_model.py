import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
import logging
import argparse
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def load_data(self, csv_path, text1_col='text1', text2_col='text2'):
        df = pd.read_csv(csv_path)
        self.text_pairs = list(zip(df[text1_col],df[text2_col]))
        logger.info(f"Loaded {len(self.text_pairs)} text pairs from {csv_path}")
        return self
    
    def prepapre_data(self, test_size=0.2):
        examples = [InputExample(texts=[text1,text2]) for text1,text2 in self.text_pairs]
        self.train_examples, self.val_examples = train_test_split(examples, test_size=test_size, random_state=42)
        return self
    
    def train(self, output_dir = 'models', epochs=3, batch_size=16):
        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        self.model.fit(train_objectives=[(train_dataloader,train_loss)],
                       epochs=epochs, 
                       warmup_steps = 100,
                       output_path=output_dir
        )
        logger.info(f"Model saved to {output_dir}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train a sentence similarity model')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the csv file containing text pairs')
    parser.add_argument('--text1_col', type=str, default='text1', help='Column name for first text')
    parser.add_argument('--text2_col', type=str, default='text2', help='Column name for second text')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    trainer = SimilarityModel()
    trainer.load_data(args.csv_path, args.text1_col, args.text2_col)
    trainer.prepapre_data()
    trainer.train(args.output_dir, args.epochs, args.batch_size)

                        
                  
        