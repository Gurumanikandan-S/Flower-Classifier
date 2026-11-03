"""
============================================
Iris Flower Classification (Beginner Project)
============================================

Goal:
    Train a machine learning model that can predict the species of an iris
    flower (Setosa, Versicolor, or Virginica) based on four measurements:
        - Sepal length (cm)
        - Sepal width  (cm)
        - Petal length (cm)
        - Petal width  (cm)

What we do in this script:
    1. Load the famous built-in Iris dataset from Scikit-learn.
    2. Explore the data a little.
    3. Split the data into a training set and a testing set.
    4. Train a Decision Tree classifier on the training set.
    5. Evaluate the model by computing its accuracy on the testing set.
    6. Let the user type in flower measurements and predict the species.

How to run:
    python main.py

Required library:
    pip install scikit-learn
============================================
"""

# ------------------------------------------------------------------
# Step 0: Import the libraries we need
# ------------------------------------------------------------------
# scikit-learn comes with the Iris dataset and the Decision Tree model.
from sklearn.datasets import load_iris     # Loads the Iris dataset
from sklearn.model_selection import train_test_split  # Splits data into train/test
from sklearn.tree import DecisionTreeClassifier       # The ML algorithm we train
from sklearn.metrics import accuracy_score            # Measures how good the model is
from sklearn.tree import export_text                  # Prints the tree in plain text


# ------------------------------------------------------------------
# Step 1: Load the dataset
# ------------------------------------------------------------------
# load_iris() returns a small object called a "Bunch".
# A Bunch works like a dictionary, but you can also use dot notation.
iris = load_iris()

# iris.data holds the 4 measurements (features) for all 150 flowers.
# Shape is (150, 4) -> 150 rows, 4 columns.
X = iris.data
print(f"Shape of feature matrix X: {X.shape}")

# iris.target holds the species for each flower, as numbers:
#   0 = Setosa, 1 = Versicolor, 2 = Virginica
y = iris.target
print(f"Shape of target vector y:   {y.shape}")

# iris.target_names is a list of the species names (labels).
# We will use this later to convert a predicted number into a name.
print(f"Species names: {list(iris.target_names)}")


# ------------------------------------------------------------------
# Step 2: Split the data into training and testing sets
# ------------------------------------------------------------------
# The test_size parameter says "use 20% of the data for testing".
# The random_state parameter fixes the random shuffle so every run
# gives the same split -> makes results reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples:  {len(X_train)}")
print(f"Testing samples:   {len(X_test)}")


# ------------------------------------------------------------------
# Step 3: Create and train the Decision Tree model
# ------------------------------------------------------------------
# DecisionTreeClassifier is a model that learns simple rules from the
# data, e.g. "if petal length <= 2.5 cm, it is a Setosa".
model = DecisionTreeClassifier(random_state=42)

# fit() is the "training" step. The model studies the training data
# and figures out the best rules to separate the three species.
model.fit(X_train, y_train)


# ------------------------------------------------------------------
# Step 4: Make predictions and measure accuracy
# ------------------------------------------------------------------
# Use the model to predict species for the testing data.
y_pred = model.predict(X_test)

# accuracy_score compares the predicted species with the true species
# and returns the fraction of correct predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy on the test set: {accuracy:.2%}")

# Bonus: show the learned rules in plain text so beginners can see
# exactly what the Decision Tree learned.
print("\nLearned rules (simplified):")
print(export_text(model, feature_names=iris.feature_names))


# ------------------------------------------------------------------
# Step 5: Interactive prediction
# ------------------------------------------------------------------
def predict_flower():
    """Ask the user for flower measurements and predict the species."""
    print("\n" + "=" * 50)
    print("  Enter the measurements of your iris flower")
    print("=" * 50)

    try:
        # Get the four measurements from the user.
        # float() turns the typed text into a decimal number.
        sepal_length = float(input("Sepal length (cm): "))
        sepal_width = float(input("Sepal width  (cm): "))
        petal_length = float(input("Petal length (cm): "))
        petal_width = float(input("Petal width  (cm): "))
    except ValueError:
        # If the user types something that is not a number, we show a
        # friendly message and stop.
        print("\nInvalid input. Please enter numbers only.")
        return

    # Build a single sample (one row) with the 4 measurements.
    # The model expects the input as a 2D array, so we wrap it in [].
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]

    # predict() returns a list with one value for our single sample.
    prediction = model.predict(sample)[0]   # Example: 1 (a number)

    # Look up the species name for the predicted number.
    species = iris.target_names[prediction]

    print("-" * 50)
    print(f"Predicted species: {species}")
    print("-" * 50)


# ------------------------------------------------------------------
# Step 6: Run everything
# ------------------------------------------------------------------
# This block only runs when we execute this file directly
# (python main.py). It prevents the interactive part from running
# if this script is imported by another script.
if __name__ == "__main__":
    # Show the accuracy one more time right at the end, then start
    # the interactive prediction loop.
    print(f"Model accuracy: {accuracy:.2%}")
    predict_flower()

    # Let the user predict as many flowers as they want.
    while True:
        try:
            again = input("\nPredict another flower? (y/n): ").strip().lower()
        except EOFError:
            # Happens when there is no more input (e.g. running in a pipe).
            # We just stop the program gracefully.
            break

        if again == "y":
            predict_flower()
        else:
            print("\nThank you for using the Iris Classifier. Goodbye!")
            break
