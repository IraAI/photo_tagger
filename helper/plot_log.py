import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import re
import sys
import extract_seconds
# import sciman.sciman as sciman


def get_line_type(line):
    """Return either 'test' or 'train' depending on line type
    """

    line_type = None
    if line.find('Train') != -1:
        line_type = 'train'
    elif line.find('Test') != -1:
        line_type = 'test'
    return line_type


def parse_log(path_to_log):
    """Parse log file
    Returns (df_train, df_test)

    df_train and df_test are pandas DataFrame with data from log
    """

    re_correct_line = re.compile('^\w+\d+')
    re_iteration = re.compile('Iteration (\d+)')
    # alexnet
    #re_output_loss = re.compile('output #\d+: loss = ([\.\d]+)')
    #re_output_acc = re.compile('output #\d+: accuracy = ([\.\d]+)')

    #googlenet
    re_output_loss = re.compile('output #\d+: loss3\/loss3 = ([\.\d]+)')
    re_output_acc = re.compile('output #\d+: loss3\/top-1 = ([\.\d]+)')

    re_lr = re.compile('lr = ([\.\d]+)')

    # Pick out lines of interest
    iteration = -1
    test_accuracy = -1
    learning_rate = float('NaN')
    acc = float('NaN')
    train_dict_list = []
    test_dict_list = []
    train_dict_names = ('NumIters', 'Loss', 'Accuracy', 'LearningRate', 'Seconds')
    test_dict_names = ('NumIters', 'Loss', 'Accuracy')

    is_test_group = False

    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        start_time = extract_seconds.get_start_time(f, logfile_year)
        for line in f:
            if not re_correct_line.match(line):
                continue
            iteration_match = re_iteration.search(line)
            if iteration_match:
                iteration = int(iteration_match.group(1))
            if iteration == -1:
                # Only look for other stuff if we've found the first iteration
                continue

            time = extract_seconds.extract_datetime_from_line(line, logfile_year)
            seconds = (time - start_time).total_seconds()

            lr_match = re_lr.search(line)
            if lr_match:
                learning_rate = float(lr_match.group(1))
            output_acc_match = re_output_acc.search(line)
            if output_acc_match:
                acc = float(output_acc_match.group(1))

            output_loss_match = re_output_loss.search(line)
            if output_loss_match:
                if get_line_type(line) == 'test':
                    test_loss = float(output_loss_match.group(1))
                    test_dict_list.append({'NumIters': iteration,
                                           'Loss': test_loss,
                                           'Accuracy': acc})
                else:
                    train_loss = float(output_loss_match.group(1))
                    train_dict_list.append({'NumIters': iteration,
                                            'Loss': train_loss,
                                            'Accuracy': acc,
                                            'LearningRate': learning_rate,
                                            'Seconds': seconds})

    df_train = pd.DataFrame(columns=train_dict_names)
    df_test = pd.DataFrame(columns=test_dict_names)

    for col in train_dict_names:
        df_train[col] = [d[col] for d in train_dict_list]
    for col in test_dict_names:
        df_test[col] = [d[col] for d in test_dict_list]

    return df_train, df_test


def compose_single_log(df_train, df_test):
    df_train.columns = [c + "Train" if c != 'NumIters' else c for c in df_train.columns]
    df_test.columns = [c + "Test" if c != 'NumIters' else c for c in df_test.columns]
    df_train.index = df_train['NumIters']
    df_test.index = df_test['NumIters']
    df_train = df_train.drop('NumIters', axis=1)
    df_test = df_test.drop('NumIters', axis=1)
    df = df_train.join(df_test)
    return df


def plot_charts(train_df, test_df, root_dir, skip_train, skip_test, is_show=False, is_add_to_sciman=False):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_title(root_dir)

    is_accuracy = True

    # # Commonly value of the loss which is calculated for NN with initial randomly initialized weights
    # # is very big compared to NN after updates.
    # # So remove first points from train/test line in order to highlight "more interesting" part of chart
    # train_df = train_df.drop(range(skip_train))
    # test_df = test_df.drop(range(skip_test))

    colors = {"Train": "red", "Test": "blue", "TrainAcc": "green", "TestAcc": "black"}

    loss_train_min = train_df["Loss"].min()
    loss_train_min_iter = train_df["NumIters"][train_df["Loss"].argmin()]
    if is_accuracy:
        acc_train_max = train_df["Accuracy"].max()
        acc_train_max_iter = train_df["NumIters"][train_df["Accuracy"].argmax()]
    ax.plot(train_df["NumIters"], train_df["Loss"], color=colors["Train"],
            label="Train Loss, min={0:.4f}, it={1}".format(loss_train_min, loss_train_min_iter))
    if is_accuracy:
        ax.plot(train_df["NumIters"], train_df["Accuracy"], color=colors["TrainAcc"],
                label="Train Accuracy, max={0:.4f}, it={1}".format(acc_train_max, acc_train_max_iter))
    ax.plot(loss_train_min_iter, loss_train_min, color=colors["Train"], marker="o")
    if is_accuracy:
        ax.plot(acc_train_max_iter, acc_train_max, color=colors["TrainAcc"], marker="o")

    loss_test_min = None
    loss_test_min_iter = None
    if test_df.shape[0] > 0:
        loss_test_min = test_df["Loss"].min()
        loss_test_min_iter = test_df["NumIters"][test_df["Loss"].argmin()]
        if is_accuracy:
            acc_test_max = test_df["Accuracy"].max()
            acc_test_max_iter = test_df["NumIters"][test_df["Accuracy"].argmax()]
        ax.plot(test_df["NumIters"], test_df["Loss"], color=colors["Test"],
                label="Test Loss, min={0:.4f}, it={1}".format(loss_test_min, loss_test_min_iter))
        if is_accuracy:
            ax.plot(test_df["NumIters"], test_df["Accuracy"], color=colors["TestAcc"],
                    label="Test Accuracy, max={0:.4f}, it={1}".format(acc_test_max, acc_test_max_iter))
        ax.plot(loss_test_min_iter, loss_test_min, color=colors["Test"], marker="o")
        if is_accuracy:
            ax.plot(acc_test_max_iter, acc_test_max, color=colors["TestAcc"], marker="o")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, framealpha=0)
    plt.grid()
    plt.show()
    #if is_show:
    #    plt.show()
    #else:
    plt.savefig(os.path.join(root_dir, "model_log.png"))

    df = compose_single_log(train_df.copy(), test_df.copy())
    df.to_csv(os.path.join(root_dir, "log.csv"), index_label='Iters')

    if is_add_to_sciman:
        ds_name = root_dir.replace(re.findall(r'([0-9_-]+/{0,1}$)', root_dir)[0], '').replace("results/", "")
        print ds_name
        if root_dir.endswith('/'):
            root_dir = root_dir[:-1]
        files = [os.path.join(root_dir, "log.csv"),
                 os.path.join(root_dir, "solver.prototxt"),
                 os.path.join(root_dir, "train.prototxt"),
                 os.path.join(root_dir, "predict.prototxt")]
        imgs_fns = [os.path.join(root_dir,  fn) for fn in os.listdir(root_dir) if fn.endswith('.png') or fn.endswith('.bmp')]
        files += imgs_fns
        sciman.push_results("http://106.125.45.7/", "obaiev", root_dir, ds_name,
                            "Oleksandr Baiev", "logloss", loss_test_min, files)

    print "\nLast iter: {0}".format(train_df["NumIters"].iloc[-1])
    print "Time spent (sec): {0}\n".format(train_df["Seconds"].iloc[-1])
    print "Best Loss"
    print "\ttrain {0} on iter {1}".format(loss_train_min, loss_train_min_iter)
    print "\ttest  {0} on iter {1}".format(loss_test_min, loss_test_min_iter)
    if is_accuracy:
        print "Best Accuracy"
        print "\ttrain {0} on iter {1}".format(acc_train_max, acc_train_max_iter)
        print "\ttest  {0} on iter {1}\n".format(acc_test_max, acc_test_max_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="path to results of training")
    parser.add_argument("-r", "--skip_train", default=2, type=int, help="number of points to skip in train log")
    parser.add_argument("-e", "--skip_test", default=2, type=int, help="number of points to skip in train log")
    parser.add_argument("-a", "--add_to_sciman", default=False, type=bool, help="add result to the SciMan log server")
    args = parser.parse_args()

    df_train, df_test = parse_log(os.path.join(args.root, "nohup.out"))

    plot_charts(df_train, df_test, args.root, args.skip_train, args.skip_test, True, args.add_to_sciman)
