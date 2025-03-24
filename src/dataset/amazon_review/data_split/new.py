import os
import random
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

domain_list = ['All_Beauty', 'Baby_Products', 'Video_Games']

ori_file_dir = 'data/amazon_review/processed'
output_dir = 'data/amazon_review/processed_filtered'

for domain in domain_list:
    # Define input and output directories
    input_dir = os.path.join(ori_file_dir, domain)
    output_dir_base = output_dir
    transductive_dir = os.path.join(output_dir_base, 'transductive', domain)
    inductive_dir = os.path.join(output_dir_base, 'inductive', domain)
    os.makedirs(transductive_dir, exist_ok=True)
    os.makedirs(inductive_dir, exist_ok=True)

    # Load data and filter out rows with item history length > 10
    def load_data(filename):
        path = os.path.join(input_dir, filename)
        df = pd.read_csv(path, sep='\t', header=None, names=['user_id:token', 'item_id_list:token_seq', 'item_id:token'])
        # df = df[df['item_id_list:token_seq'].apply(lambda x: len(str(x).split()) <= 10)]
        return df

    # Load datasets
    train_df = load_data(f'{domain}.train.inter')
    valid_df = load_data(f'{domain}.valid.inter')
    test_df = load_data(f'{domain}.test.inter')
    
    # Sample 50,000 rows from training data
    train_df = train_df.sample(n=min(50000, len(train_df)), random_state=42)
    train_users = set(train_df['user_id:token'])

    # Split valid/test data into transductive and inductive based on user presence in train
    def split_transductive_inductive(df, total=1000):
        transductive = df[df['user_id:token'].isin(train_users)].sample(n=total // 2, random_state=42)
        inductive = df[~df['user_id:token'].isin(train_users)].sample(n=total // 2, random_state=42)
        return transductive, inductive
    
    # Split validation and test datasets
    valid_trans, valid_ind = split_transductive_inductive(valid_df, 1000)
    test_trans, test_ind = split_transductive_inductive(test_df, 1000)

    # Save datasets to respective folders
    train_df.to_csv(os.path.join(transductive_dir, f'{domain}.train.inter'), sep='\t', header=True, index=False)

    valid_trans.to_csv(os.path.join(transductive_dir, f'{domain}.valid.inter'), sep='\t', header=True, index=False)
    test_trans.to_csv(os.path.join(transductive_dir, f'{domain}.test.inter'), sep='\t', header=True, index=False)

    train_df.to_csv(os.path.join(inductive_dir, f'{domain}.train.inter'), sep='\t', header=True, index=False)
    valid_ind.to_csv(os.path.join(inductive_dir, f'{domain}.valid.inter'), sep='\t', header=True, index=False)
    test_ind.to_csv(os.path.join(inductive_dir, f'{domain}.test.inter'), sep='\t', header=True, index=False)

    print("Processing completed. Data saved in:")
    print("Transductive:", transductive_dir)
    print("Inductive:", inductive_dir)
