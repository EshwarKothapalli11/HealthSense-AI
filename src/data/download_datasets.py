"""
download_datasets.py — Downloads and saves all three datasets for HealthSense AI.

Handles:
  - Pima Indians Diabetes dataset (CSV from GitHub)
  - Heart Disease dataset (UCI ML Repository)
  - Mental Health text dataset (HuggingFace / GitHub, with rich synthetic fallback)
"""

import os
import random
import pandas as pd
import requests
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def download_diabetes() -> pd.DataFrame:
    """
    Download the Pima Indians Diabetes dataset.

    Returns:
        pd.DataFrame: The diabetes dataset with proper column headers.
    """
    print("[INFO] Downloading Pima Indians Diabetes dataset...")
    df = pd.read_csv(config.DIABETES_URL, header=None, names=config.DIABETES_COLUMNS)
    _ensure_dir("data/raw")
    df.to_csv("data/raw/diabetes.csv", index=False)
    print(f"  -> Diabetes dataset saved -- shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


def download_heart() -> pd.DataFrame:
    """
    Download the Heart Disease dataset from UCI ML Repository.

    Returns:
        pd.DataFrame: The heart disease dataset with binarized target.
    """
    print("[INFO] Downloading Heart Disease dataset (UCI ID=45)...")
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        X = heart.data.features
        y = heart.data.targets
        df = pd.concat([X, y], axis=1)

        # Binarize target: num >= 1 -> 1, else 0
        target_col = y.columns[0]
        df['target'] = (df[target_col] >= 1).astype(int)
        if target_col != 'target':
            df = df.drop(columns=[target_col])
    except Exception as e:
        print(f"  [WARN] UCI fetch failed ({e}). Using fallback URL...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
        ]
        df = pd.read_csv(url, header=None, names=cols, na_values='?')
        df = df.dropna()
        df['target'] = (df['num'] >= 1).astype(int)
        df = df.drop(columns=['num'])

    _ensure_dir("data/raw")
    df.to_csv("data/raw/heart.csv", index=False)
    print(f"  -> Heart Disease dataset saved -- shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


def _generate_synthetic_mental_health(n: int = 8000) -> pd.DataFrame:
    """
    Generate high-quality synthetic mental health text data as fallback.

    Creates diverse, realistic text samples with strong class separation
    to enable effective LSTM training even without real data.

    Args:
        n: Number of rows to generate.

    Returns:
        pd.DataFrame: Synthetic dataset with 'text' and 'label' columns.
    """
    random.seed(42)

    # --- DEPRESSIVE / STRESSED phrases (label=1) ---
    depressive_starters = [
        "I feel so hopeless and empty inside, like nothing will ever change",
        "Nothing brings me joy anymore, everything feels grey and meaningless",
        "I can't stop crying for no reason, the tears just keep coming",
        "I feel like a burden to everyone around me, they'd be better off without me",
        "I don't see the point in anything anymore, life feels hollow",
        "Every day feels like a struggle just to get through the basic tasks",
        "I feel completely alone even when surrounded by people who care",
        "I can't concentrate on anything, my mind is always foggy and confused",
        "I've lost interest in all the things I used to love doing",
        "I feel worthless and like nothing will ever get better for me",
        "I can't sleep at night, my anxious thoughts won't stop racing",
        "I feel exhausted all the time no matter how much I rest or sleep",
        "I've been having dark thoughts about my future and what lies ahead",
        "I feel disconnected from everyone and everything in my life",
        "I don't have the energy to do even the simplest daily tasks",
        "I feel like I'm drowning in an ocean of sadness and despair",
        "I've been isolating myself from friends and family for weeks now",
        "I feel anxious and stressed about every little thing in my life",
        "I can't remember the last time I felt truly happy or content",
        "I feel trapped in my own mind with no way out of this darkness",
        "Life feels meaningless and I'm just going through the motions every day",
        "I wake up dreading each new day and wishing I could stay in bed forever",
        "My appetite has completely disappeared and I barely eat anymore",
        "I feel numb and unable to feel any positive emotions at all",
        "I can't stop worrying about things that are completely out of my control",
        "The weight of depression makes every moment feel unbearable",
        "I feel like a failure in everything I try to accomplish",
        "My self-esteem is at rock bottom and I hate who I've become",
        "I'm scared of the future because I can't imagine things getting better",
        "I feel overwhelmed by even the smallest responsibilities in my life",
        "Panic attacks have been consuming me, I can barely breathe sometimes",
        "I feel so lonely it physically hurts, the isolation is crushing me",
        "I've lost all motivation, can't even force myself to get out of bed",
        "Negative thoughts spiral in my head constantly, they never stop",
        "I feel broken beyond repair, like something fundamental is wrong with me",
        "The sadness is so heavy it feels like I'm carrying the entire world",
        "I can't find meaning or purpose in anything I do anymore",
        "My mental health has deteriorated so much I barely recognize myself",
        "I feel helpless watching my life fall apart around me",
        "Every night I lie awake consumed by regret and self-loathing",
        "I'm struggling with severe anxiety that controls every decision I make",
        "The constant stress is destroying my health and my relationships",
        "I feel so depressed I can barely function at work or school",
        "My mind is a prison of negative thoughts I can never escape from",
        "I feel invisible, like nobody notices or cares about my suffering",
        "I've been crying myself to sleep every night for the past month",
        "I can't shake this persistent feeling of dread and hopelessness",
        "I feel emotionally drained and have nothing left to give anyone",
        "The darkness inside me just keeps growing and spreading",
        "I'm exhausted from pretending to be okay when I'm falling apart inside",
    ]

    depressive_additions = [
        "I wish I could just disappear and not deal with any of this.",
        "The pain inside me is unbearable most days.",
        "I don't know how much longer I can keep going like this.",
        "It feels like nobody truly understands what I'm going through.",
        "I've tried everything and nothing seems to help me feel better.",
        "My therapist says I need to be patient but I'm running out of patience.",
        "I can barely function at work because of how depressed I feel.",
        "The medication doesn't seem to be helping at all anymore.",
        "I feel guilty for feeling this way when others have it worse.",
        "Sometimes I wonder if things will ever be normal again.",
        "I've been having trouble eating and sleeping for weeks now.",
        "The world feels like a dark and hostile place to me.",
        "I feel like screaming but I know nobody would hear me.",
        "My relationships are suffering because of my mental state.",
        "I can't stop the intrusive thoughts no matter how hard I try.",
        "Every small setback feels catastrophic and overwhelming to me.",
        "I've withdrawn from all social activities because I can't cope.",
        "The anxiety makes my heart race and my hands shake constantly.",
        "I feel paralyzed by fear and indecision every single day.",
        "My emotions are so volatile, I go from numb to overwhelmed instantly.",
    ]

    # --- HEALTHY / NEUTRAL phrases (label=0) ---
    healthy_starters = [
        "I had a wonderful and productive day at work today, feeling accomplished",
        "Enjoyed a beautiful walk in the park this morning, the weather was perfect",
        "Looking forward to the weekend plans with my friends and family",
        "Just finished reading a really inspiring book, highly recommend it to everyone",
        "Had an amazing productive meeting and feeling very accomplished today",
        "Spent quality time with family over a delicious homemade dinner",
        "The weather was beautiful today, perfect for outdoor activities and exercise",
        "Started learning a new hobby and I'm really enjoying every moment of it",
        "Feeling grateful for the incredible support system of my friends and family",
        "Had a fantastic workout at the gym this morning, feeling strong and energized",
        "Cooked a delicious healthy meal and shared it with my roommates tonight",
        "Planning an exciting vacation and feeling so thrilled about the adventure ahead",
        "Got some amazing positive feedback on my project at work today",
        "Feeling so relaxed and centered after a peaceful yoga session this evening",
        "Enjoyed a wonderful movie night with popcorn, laughter, and great company",
        "Feeling energized, motivated, and ready to tackle new exciting challenges",
        "Had a fascinating deep conversation with a colleague about innovative ideas",
        "The sunset was absolutely stunning and breathtaking this evening",
        "Tried a new restaurant downtown and the food was absolutely incredible",
        "Feeling content, grateful, and at peace with where I am in my life",
        "Played sports with friends today and had an absolute blast outdoors",
        "Reading a fascinating article about exciting new scientific discoveries",
        "Organized my entire workspace and I'm feeling so much more focused now",
        "Volunteered at the community center today, it was incredibly rewarding",
        "Celebrated a meaningful personal achievement today, it felt really amazing",
        "Woke up feeling refreshed and excited about all the possibilities for today",
        "Had a great catch-up call with old friends, we laughed so much together",
        "Finished a challenging puzzle and felt so proud of my determination",
        "Went on an incredible hike today and the views from the top were spectacular",
        "Feeling blessed and thankful for all the good things happening in my life",
        "Made significant progress on my personal goals today, feeling on track",
        "Spent the afternoon gardening and it was so therapeutic and calming",
        "Had the most delightful coffee chat with my best friend this morning",
        "Learning to play guitar and I can already see myself improving every day",
        "Completed my morning meditation routine and feeling centered and grounded",
        "The kids made me laugh so hard today with their silly creative jokes",
        "Enjoying a peaceful quiet evening at home with a warm cup of tea",
        "Feeling strong and confident about the direction my career is heading",
        "Made someone's day by doing a random act of kindness, it felt wonderful",
        "Grateful for good health, amazing friends, and exciting opportunities ahead",
        "Today was one of those rare perfect days where everything just clicked",
        "I'm so excited about the new project I'm working on, the possibilities are endless",
        "Feeling optimistic and hopeful about the future and all it holds for me",
        "Had the best sleep last night and woke up feeling completely rejuvenated",
        "Discovered a beautiful new hiking trail and can't wait to explore more of it",
        "My hard work is finally paying off and I'm proud of how far I've come",
        "Spent the day at the beach with family, soaking up sun and making memories",
        "Feeling inspired after attending an incredible motivational workshop today",
        "Life is treating me well and I appreciate every single moment of it",
        "Accomplished everything on my to-do list and treated myself to ice cream",
    ]

    healthy_additions = [
        "I'm really looking forward to what tomorrow brings.",
        "Life feels full of exciting possibilities right now.",
        "I feel so fortunate to have such wonderful people in my life.",
        "Today reminded me of how beautiful the simple things can be.",
        "I'm grateful for another amazing day full of joy and laughter.",
        "There's so much to look forward to in the coming weeks.",
        "I feel strong, healthy, and capable of achieving anything.",
        "The positive energy around me is contagious and uplifting.",
        "I love the routine I've built, it keeps me balanced and happy.",
        "My confidence has been growing steadily and it feels great.",
        "Every day I'm getting closer to the person I want to become.",
        "The support from my community means the world to me.",
        "I feel lucky to be alive and experiencing this beautiful world.",
        "My relationships are thriving and bringing me so much happiness.",
        "I'm in a really good place mentally and emotionally right now.",
        "I've found inner peace through self-care and mindful practices.",
        "The love I receive from family and friends sustains me daily.",
        "Feeling absolutely amazing about my personal growth journey.",
        "Each morning brings new opportunities to learn and improve.",
        "I'm living my best life and enjoying every single second of it.",
    ]

    modifiers_dep = [
        "Lately,", "Recently,", "These days,", "For weeks now,",
        "I don't know why but", "It's been getting worse and",
        "I tried everything but", "No matter what I do,",
        "Unfortunately,", "Sadly,", "To be honest,",
        "I hate to admit it but", "I'm ashamed to say",
        "Honestly,", "The truth is,",
    ]

    modifiers_healthy = [
        "Today,", "This morning,", "Just now,", "Earlier today,",
        "This week,", "Yesterday,", "Over the weekend,",
        "Happily,", "Thankfully,", "Fortunately,",
        "I'm so glad that", "I'm thrilled because",
        "I'm happy to share that", "Excitingly,",
        "I'm pleased to say", "Wonderfully,",
    ]

    texts = []
    labels = []

    for _ in range(n):
        if random.random() < 0.5:
            # Generate depressive text
            starter = random.choice(depressive_starters)
            addition = random.choice(depressive_additions)
            modifier = random.choice(modifiers_dep)

            # Vary the text construction for diversity
            roll = random.random()
            if roll < 0.30:
                text = f"{modifier} {starter.lower()}. {addition}"
            elif roll < 0.55:
                text = f"{starter}. {addition}"
            elif roll < 0.75:
                text = f"{modifier} {starter.lower()}"
            else:
                text = f"{addition} {starter.lower()}"

            labels.append(1)
        else:
            # Generate healthy text
            starter = random.choice(healthy_starters)
            addition = random.choice(healthy_additions)
            modifier = random.choice(modifiers_healthy)

            roll = random.random()
            if roll < 0.30:
                text = f"{modifier} {starter.lower()}. {addition}"
            elif roll < 0.55:
                text = f"{starter}. {addition}"
            elif roll < 0.75:
                text = f"{modifier} {starter.lower()}"
            else:
                text = f"{addition} {starter.lower()}"

            labels.append(0)

        texts.append(text)

    return pd.DataFrame({'text': texts, 'label': labels})


def download_mental_health() -> pd.DataFrame:
    """
    Download mental health text dataset.

    Attempts multiple sources:
      1. HuggingFace sentiment-analysis-for-mental-health dataset (53K rows)
      2. GitHub mental health CSV
      3. Rich synthetic fallback (8000 samples)

    Returns:
        pd.DataFrame: Text dataset with 'text' and 'label' columns.
    """
    print("[INFO] Downloading Mental Health text dataset...")
    _ensure_dir("data/raw")

    # --- Attempt 1: HuggingFace CSV ---
    hf_url = (
        "https://huggingface.co/datasets/AhmedSSoliman/"
        "sentiment-analysis-for-mental-health-Combined-Data/"
        "resolve/main/sentiment-analysis-for-mental-health-Combined%20Data.csv"
    )
    try:
        print("  Trying HuggingFace dataset...")
        response = requests.get(hf_url, timeout=30)
        response.raise_for_status()

        with open("data/raw/mental_health_raw.csv", 'wb') as f:
            f.write(response.content)
        df = pd.read_csv("data/raw/mental_health_raw.csv")

        # This dataset has columns: 'statement' and 'status'
        # Rename to standard format
        text_col = None
        label_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ('statement', 'text', 'content', 'message'):
                text_col = c
            elif cl in ('status', 'label', 'class', 'sentiment', 'category'):
                label_col = c

        if text_col and label_col:
            df = df[[text_col, label_col]].dropna()
            df.columns = ['text', 'status']

            # Binarize: any mental health condition -> 1, Normal -> 0
            normal_variants = ['normal', 'healthy', 'no', '0', 'none']
            df['label'] = df['status'].apply(
                lambda x: 0 if str(x).strip().lower() in normal_variants else 1
            )
            df = df[['text', 'label']]
            df.to_csv("data/raw/mental_health.csv", index=False)
            print(f"  -> HuggingFace Mental Health dataset saved -- shape: {df.shape}")
            print(f"  Label distribution: {dict(df['label'].value_counts())}")
            return df

    except Exception as e:
        print(f"  [WARN] HuggingFace download failed: {e}")

    # --- Attempt 2: Original GitHub URL ---
    try:
        print("  Trying GitHub dataset...")
        response = requests.get(config.MENTAL_HEALTH_URL, timeout=15)
        response.raise_for_status()

        with open("data/raw/mental_health_raw.csv", 'w', encoding='utf-8') as f:
            f.write(response.text)
        df = pd.read_csv("data/raw/mental_health_raw.csv")

        if 'text' not in df.columns:
            text_cols = [c for c in df.columns if 'text' in c.lower() or 'statement' in c.lower()]
            if text_cols:
                df = df.rename(columns={text_cols[0]: 'text'})

        if 'label' not in df.columns:
            label_cols = [c for c in df.columns
                         if 'label' in c.lower() or 'status' in c.lower() or 'class' in c.lower()]
            if label_cols:
                col = label_cols[0]
                unique_labels = df[col].unique()
                if len(unique_labels) > 2:
                    normal_variants = ['normal', 'healthy', 'no', '0']
                    df['label'] = df[col].apply(
                        lambda x: 0 if str(x).strip().lower() in normal_variants else 1
                    )
                else:
                    df = df.rename(columns={col: 'label'})

        df = df[['text', 'label']].dropna()
        df.to_csv("data/raw/mental_health.csv", index=False)
        print(f"  -> GitHub Mental Health dataset saved -- shape: {df.shape}")
        return df

    except Exception as e:
        print(f"  [WARN] GitHub download failed: {e}")

    # --- Attempt 3: Rich synthetic fallback ---
    print("  Generating rich synthetic mental health data (8000 samples)...")
    df = _generate_synthetic_mental_health(8000)
    df.to_csv("data/raw/mental_health.csv", index=False)
    print(f"  -> Synthetic Mental Health dataset saved -- shape: {df.shape}")
    print(f"  Label distribution: {dict(df['label'].value_counts())}")
    return df


def download_all() -> dict[str, pd.DataFrame]:
    """
    Download all three datasets.

    Returns:
        dict: Dictionary mapping dataset names to DataFrames.
    """
    print("=" * 60)
    print("  HealthSense AI -- Dataset Download Pipeline")
    print("=" * 60)

    datasets = {}
    datasets['diabetes'] = download_diabetes()
    datasets['heart'] = download_heart()
    datasets['mental_health'] = download_mental_health()

    print("\n" + "=" * 60)
    print("  All datasets downloaded successfully!")
    print("=" * 60)

    return datasets


if __name__ == "__main__":
    download_all()
