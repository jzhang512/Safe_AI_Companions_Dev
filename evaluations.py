import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

def calculate_basic_stats(prediction_file_name, category=None):
    """
        Calculates basic statistics such as:
        - total number of predictions under this category
        - number of e.d. onsets under this category
        - average distance of how far off the predictions are (MAE)
        - standard deviation
        - mean square error (MSE)
        - root mean square error (RMSE)
        - accuracy
        - precision
        - recall
        - f1 score
            TP: there is e.d. & classifier predicts e.d.
            FP: there is no e.d. & classifier predicts there is e.d
            FN: there is e.d. & but classifier predicts no e.d.
            TN: no e.d. & classifier predicts no e.d.

        Args:
           prediction_file_name (str): path to file with predictions (jsonl format)
           category (str): specific category from user age group to chatbot function
    """

    with open(prediction_file_name, "r") as f:
        total_distance = 0
        distances = []
        mse_total = 0
        num_total = 0
        num_ed = 0

        accuracy_count = 0
        tp_count = 0
        fp_count = 0
        fn_count = 0
        tn_count = 0

        for line in f:
            data = json.loads(line)
            age_group = data["user_age"]
            chatbot_function = data["chatbot_function"]

            if category is not None and (age_group != category and chatbot_function != category):
                continue

            onset_round_predict = data["onset_round_predict"]
            onset_round_gt = data["onset_round_gt"]

            if onset_round_gt == onset_round_predict:
                accuracy_count += 1
            
            num_total += 1
            
            # No emotional dependence.
            if onset_round_gt == -1:

                if onset_round_predict != -1:
                    fp_count += 1
                else:
                    tn_count += 1

                continue    # skip

            # Metrics including only "e.d." below.

            num_ed += 1

            if onset_round_predict != -1:
                tp_count += 1

            if onset_round_predict == -1:
                fn_count += 1

            distance = abs(onset_round_predict - onset_round_gt)
            distances.append(distance)
            squared_distance = distance ** 2

            mse_total += squared_distance
            total_distance += distance

        average_distance = total_distance / num_ed if num_ed > 0 else 0
        mse = mse_total / num_ed if num_ed > 0 else 0
        rmse = mse ** 0.5
        standard_deviation = np.std(distances)
        
        accuracy = accuracy_count / num_total if num_total > 0 else 0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "num_total": num_total,
            "num_ed": num_ed,
            "category": category,
            "average_distance": average_distance, 
            "standard_deviation": float(standard_deviation),
            "mse": mse, 
            "rmse": rmse,
            "accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "f1_score": f1_score,
            "true_positives": tp_count,
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "true_negatives": tn_count
        }

def plot_confusion_matrix(tp, fp, fn, tn):
    """
        Heatmap of confusion matrix.
    """
    conf_matrix = np.array([[tp, fp], [fn, tn]])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Onset', 'No Onset'], yticklabels=['Onset', 'No Onset'], annot_kws={"size": 24} )
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Emotional Dependence Detection Matrix')
    plt.show()

def visualize_displacement(prediction_file_name):  
    """
        Plots a graph showing all the predictions.
            x-axis: onset round
            y-axis: displacement
    """

    with open(prediction_file_name, "r") as f:
        onset_values = []
        displacement_values = []

        for line in f:
            data = json.loads(line)
            onset_round_predict = data["onset_round_predict"]
            onset_round_gt = data["onset_round_gt"]

            if onset_round_gt != -1:
                onset_values.append(onset_round_gt)
                displacement_values.append(onset_round_predict - onset_round_gt)

        # Define the range for x (matching your data's x-axis range)
    x_range = np.linspace(min(onset_values), max(onset_values), 500)

    # Compute y = -x for the defined range
    y_values = -x_range

    # Add the line to the plot
    plt.plot(x_range, y_values, color='red', linewidth=1.5, label='y = -x', alpha=0.3)

    sns.scatterplot(x=onset_values, y=displacement_values, alpha=0.15, color='blue')
    plt.xlabel("Actual Onset Round")
    plt.ylabel("Displacement from Ground Truth") 
    plt.title("Emotional Dependence Onset Round Predictions")

    
    plt.show()

def create_pie_charts(prediction_file_name):
    """
        Creates pie charts to visualize distribution of age groups and chatbot types.
    """

    with open(prediction_file_name, "r") as f:
        age_groups = []
        chatbot_types = []

        for line in f:
            data = json.loads(line)
            age_group = data["user_age"]
            chatbot_type = data["chatbot_function"]

            age_groups.append(age_group)
            chatbot_types.append(chatbot_type)

        age_group_counts = Counter(age_groups)
        chatbot_type_counts = Counter(chatbot_types)

        desired_order = ["teen", "70s", "60s", "50s", "40s", "30s", "20s"]
        ordered_counts = {group: age_group_counts.get(group, 0) for group in desired_order}

        age_group_labels = ordered_counts.keys()
        age_group_values = ordered_counts.values()

        chatbot_type_labels = chatbot_type_counts.keys()
        chatbot_type_values = chatbot_type_counts.values()

        color_palette = sns.color_palette("pastel")

        plt.pie(age_group_values, labels=age_group_labels, autopct='%1.1f%%', colors=color_palette[:len(age_group_labels)], textprops={"fontsize": 14})
        plt.title("Distribution of User Age Groups")
        plt.show()

        plt.pie(chatbot_type_values, labels=chatbot_type_labels, autopct='%1.1f%%', colors=color_palette[len(age_group_labels):len(age_group_labels) + len(chatbot_type_labels)], textprops={"fontsize": 14})
        plt.title("Distribution of Companion Functions")
        plt.show()

def histogram_onset_distribution(prediction_file_name):
    """
        Creates a histogram to visualize the distribution of onset rounds.
    """
    with open(prediction_file_name, "r") as f:
        onset_values = []

        for line in f:
            data = json.loads(line)
            onset_round_gt = data["onset_round_gt"]

            onset_values.append(onset_round_gt)

        onset_values.sort()
        sns.histplot(onset_values, bins=31, edgecolor='black', color='blue', alpha = 0.4)

        # Set x-axis ticks every 5 rounds.
        max_onset = max([v for v in onset_values if isinstance(v, (int, float))]) 
        plt.xticks(range(0, int(max_onset) + 1, 5)) 

        plt.xlabel("Onset Round")
        plt.ylabel("Frequency")
        plt.title("Distribution of Onset Rounds")
        plt.show()

def message_length_calculation(dataset_file_name):
    """
        Returns average and standard deviation of message length (user and companion) in generated dataset.

        Args:
            dataset_file_name (str): The filename of the dataset file (assumes jsonl format).
    """
    with open(dataset_file_name, "r") as f:

        user_message_lengths = []
        companion_message_lengths = []

        for line in f:
            data = json.loads(line)
            chat_history = data["chat_history"][3:]

            for message in chat_history:
                if message["role"] == "user":   # user is companion (flipped)
                    companion_message_lengths.append(len(message["content"].split()))
                else:
                    user_message_lengths.append(len(message["content"].split()))

        user_avg_length = np.mean(user_message_lengths)
        companion_avg_length = np.mean(companion_message_lengths)
        user_std_dev = np.std(user_message_lengths)
        companion_std_dev = np.std(companion_message_lengths)

        return {
            "user_avg_length": user_avg_length,
            "user_std_dev": user_std_dev,
            "companion_avg_length": companion_avg_length,
            "companion_std_dev": companion_std_dev
        }

def message_topic_modelling(dataset_file_name, chatbot_function, model_save_path):
    """
        Use Gensim library to perform LDA topic modelling on conversations by chatbot function. Prints out topics.

        Args:
            dataset_file_name (str): The filename of the dataset file (assumes jsonl format).
            chatbot_function (str): The chatbot function to filter for (must be one of "friend", "significant other", "family member").
            model_save_path (str): The path to save the trained LDA model.
    """

    messages = []

    with open(dataset_file_name, "r") as f:
        for line in f:
            data = json.loads(line)

            if data["chatbot_function"] != chatbot_function:
                continue

            chat_history = data["chat_history"][3:]

            for message in chat_history:
                if message["role"] == "assistant":   # user is companion (flipped)
                    messages.append(message["content"])

    # Preprocess messages.
    stop_words = set(stopwords.words("english"))
    processed_messages = [
    [word for word in simple_preprocess(msg) if word not in stop_words]
    for msg in messages
    ]

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(processed_messages)
    corpus = [dictionary.doc2bow(message) for message in processed_messages]

    # Train the LDA model
    num_topics = 5
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)

     # Save the model, dictionary, and corpus
    lda_model.save(f"{model_save_path}.gensim")
    dictionary.save(f"{model_save_path}_dictionary.gensim")
    with open(f"{model_save_path}_corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)

    print(f"LDA model, dictionary, and corpus saved to '{model_save_path}'.")

    # Step 5: Print the topics
    topics = lda_model.print_topics(num_topics=num_topics, num_words=5)
    for idx, topic in topics:
        print(f"Topic {idx}: {topic}")

def print_topics(model_saved_path):
    """
        Prints topics from a saved trained LDA model.

        Args:
            model_save_path (str): The path to the saved model.
    """
    print(model_saved_path)
    lda_model = LdaModel.load(f"{model_saved_path}.gensim")
    topics = lda_model.print_topics(num_topics=5, num_words=5)
    for idx, topic in topics:
        print(f"Topic {idx}: {topic}")
    print()

if __name__ == "__main__":

    categories = [None, "teen", "20s", "30s", "40s", "50s", "60s", "70s", "significant other", "family member", "friend"]

    for category in categories:
        print(calculate_basic_stats("prediction_final.jsonl", category = category))
    
    #plot_confusion_matrix(246,35,0,19)

    #visualize_displacement("prediction_final.jsonl")

    #create_pie_charts("prediction_final.jsonl")

    #histogram_onset_distribution("prediction_final.jsonl")

    #print(message_length_calculation("dataset_final.jsonl"))

    # chatbot_functions = ["friend", "significant other", "family member"]

    # for function in chatbot_functions:
    #     save_names = function.replace(" ", "_") + "_model"
    #     message_topic_modelling("dataset_final.jsonl", function, save_names)

    # model_names = ["significant_other_model", "family_member_model", "friend_model"]

    # for model in model_names:
    #     print_topics("./topic_modelling/" + model)
    