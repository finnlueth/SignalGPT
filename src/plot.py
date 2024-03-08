import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix_plot(df_cm, decoding, title="Confusion Matrix"):
    plt.figure(figsize=(16, 9))
    sns.set(font_scale=2)

    ax = sns.heatmap(
        df_cm,
        annot=True,
        xticklabels=[decoding[label] for label in range(len(decoding))],
        yticklabels=[decoding[label] for label in range(len(decoding))],
        fmt="d",
        label="big",
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return ax


def loss_plot(df_log):
    plt.figure(figsize=(16, 9))
    sns.lineplot(
        x=df_log.index, y=df_log["loss"], label="Training Loss", color="orange"
    )
    sns.lineplot(
        x=df_log.index, y=df_log["eval_loss"], label="Validation Loss", color="blue"
    )
    sns.lineplot(
        x=df_log.index,
        y=df_log["eval_accuracy_metric"],
        label="Accuracy",
        linestyle="--",
        color="green",
    )
    # sns.lineplot(x=df_log.index, y=df_log['eval_precision_metric'], label='Precision', linestyle=':', color='purple')
    # sns.lineplot(x=df_log.index, y=df_log['eval_recall_metric'], label='Recall', linestyle='-.', color='orange')
    # sns.lineplot(x=df_log.index, y=df_log['eval_f1_metric'], label='F1 Score', linestyle='-', color='red')
    sns.lineplot(
        x=df_log.index,
        y=df_log["eval_matthews_correlation"],
        label="Matthews Correlation",
        linestyle="--",
        color="brown",
    )

    plt.xlabel("Step")
    plt.ylabel("Metrics")
    plt.title("Training Loss and Evaluation Metrics")

    plt.legend()

    return plt


# def validation_evaluation_plots(df_training_log: pd.DataFrame, decoding):
#     # print(decoding)
#     CM = []

#     # for x in df_training_log['eval_confusion_matrix'][df_training_log['eval_confusion_matrix'].notnull()]:
#     #     cm = confusion_matrix_plot(
#     #         data_cm=x,
#     #         decoding=decoding
#     #         )
#         # CM.append(cm)

#     lp = loss_plot(df_training_log)
# return lp