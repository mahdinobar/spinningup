import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    # sns.tsplot(data=, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    # sns.lineplot(
    #     data=data,
    #     x=xaxis,  # your time column
    #     y=value,  # the value to plot
    #     hue=condition,  # group by condition
    #     units="Unit",  # repeated measures
    #     estimator=None,  # no aggregation; plot each Unit
    #     ci='sd'  # show standard deviation if you do want an aggregate
    # )

    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except Exception as error:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                print(error)
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean'):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator,zorder=1)
    # plt.hlines(44.47124, 0, 10e4, 'k', linestyles="dashed", label='PID only')
    # plt.hlines( 68.5, 0, 10e5, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 107, 0, 1.632e6, 'k', linestyles="dashed", label='PI only')
    plt.hlines( 66.03, 0, 1.632e6, 'k', linestyles="dashdot", label='PI only')
    plt.legend()
    plt.savefig(all_logdirs[-1]+"/learning_curve", format="png", bbox_inches='tight',zorder=2)
    plt.show()


    # Register the specific Times New Roman font
    times_new_roman_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    times_new_roman_font = font_manager.FontProperties(fname=times_new_roman_path)
    # Create DataFrame (assuming data[0] is available)
    df = pd.DataFrame(data[0])
    # Extract the series
    average_test_returns = df.get("AverageTestEpRet")
    # Generate epoch indices
    epochs = range(1, len(average_test_returns) + 1)
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, average_test_returns, color="blue")
    # plt.hlines( 107, 0, 4000, 'k', linestyles="dashed", label='PI only')
    plt.hlines( 66.03, 0, 4000, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 54.43, 4000, 8000, 'r', linestyles="dashed", label='PI only')
    plt.xlabel("Number of Epochs", fontproperties=times_new_roman_font, fontsize=14)
    plt.ylabel("Average Test Return", fontproperties=times_new_roman_font, fontsize=14)
    # plt.title("Average Test Return vs Epochs", fontsize=14, fontproperties=times_new_roman_font)
    plt.xticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.yticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.xlim([0,len(average_test_returns)])
    plt.grid(True)
    plt.legend(prop=times_new_roman_font, fontsize=14)
    plt.tight_layout()
    plt.savefig(all_logdirs[-1] + "/learning_curve", format="pdf")
    plt.show()

    # Register the specific Times New Roman font
    times_new_roman_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    times_new_roman_font = font_manager.FontProperties(fname=times_new_roman_path)
    # Create DataFrame (assuming data[0] is available)
    df = pd.DataFrame(data[0])
    # Extract the series
    average_test_returns = df.get("Alpha")
    # Generate epoch indices
    epochs = range(1, len(average_test_returns) + 1)
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, average_test_returns, color="blue")
    plt.xlabel("Number of Epochs", fontproperties=times_new_roman_font, fontsize=14)
    plt.ylabel("Alpha", fontproperties=times_new_roman_font, fontsize=14)
    # plt.title("Average Test Return vs Epochs", fontsize=14, fontproperties=times_new_roman_font)
    plt.xticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.yticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.xlim([0,len(average_test_returns)])
    plt.grid(True)
    plt.legend(prop=times_new_roman_font, fontsize=14)
    plt.tight_layout()
    plt.savefig(all_logdirs[-1] + "/Alpha_epoch", format="pdf")
    plt.show()

    # Register the specific Times New Roman font
    times_new_roman_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    times_new_roman_font = font_manager.FontProperties(fname=times_new_roman_path)
    # Create DataFrame (assuming data[0] is available)
    df = pd.DataFrame(data[0])
    # Extract the series
    average_test_returns = df.get("AverageLogPi")
    # Generate epoch indices
    epochs = range(1, len(average_test_returns) + 1)
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, average_test_returns, color="blue")
    plt.xlabel("Number of Epochs", fontproperties=times_new_roman_font, fontsize=14)
    plt.ylabel("AverageLogPi", fontproperties=times_new_roman_font, fontsize=14)
    # plt.title("Average Test Return vs Epochs", fontsize=14, fontproperties=times_new_roman_font)
    plt.xticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.yticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.xlim([0,len(average_test_returns)])
    plt.grid(True)
    plt.legend(prop=times_new_roman_font, fontsize=14)
    plt.tight_layout()
    plt.savefig(all_logdirs[-1] + "/AverageLogPi_epoch", format="pdf")
    plt.show()

    # Register the specific Times New Roman font
    times_new_roman_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    times_new_roman_font = font_manager.FontProperties(fname=times_new_roman_path)
    # Create DataFrame (assuming data[0] is available)
    df = pd.DataFrame(data[0])
    # Extract the series
    average_test_returns = df.get("AverageEpRet")
    # Generate epoch indices
    epochs = range(1, len(average_test_returns) + 1)
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, average_test_returns, color="blue")
    plt.xlabel("Number of Epochs", fontproperties=times_new_roman_font, fontsize=14)
    plt.ylabel("AverageEpRet", fontproperties=times_new_roman_font, fontsize=14)
    plt.hlines( 66.03, 0, len(average_test_returns), 'k', linestyles="dashed", label='PI only')
    # plt.title("Average Test Return vs Epochs", fontsize=14, fontproperties=times_new_roman_font)
    plt.xticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.yticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.xlim([0,len(average_test_returns)])
    plt.grid(True)
    plt.legend(prop=times_new_roman_font, fontsize=14)
    plt.tight_layout()
    plt.savefig(all_logdirs[-1] + "/AverageEpRet_epoch", format="pdf")
    plt.show()

    # Register the specific Times New Roman font
    times_new_roman_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    times_new_roman_font = font_manager.FontProperties(fname=times_new_roman_path)
    # Create DataFrame (assuming data[0] is available)
    df = pd.DataFrame(data[0])
    # Extract the series
    average_test_returns = df.get("LossAlpha")
    # Generate epoch indices
    epochs = range(1, len(average_test_returns) + 1)
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, average_test_returns, color="blue")
    plt.xlabel("Number of Epochs", fontproperties=times_new_roman_font, fontsize=14)
    plt.ylabel("LossAlpha", fontproperties=times_new_roman_font, fontsize=14)
    # plt.title("Average Test Return vs Epochs", fontsize=14, fontproperties=times_new_roman_font)
    plt.xticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.yticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.xlim([0,len(average_test_returns)])
    plt.grid(True)
    plt.legend(prop=times_new_roman_font, fontsize=14)
    plt.tight_layout()
    plt.savefig(all_logdirs[-1] + "/LossAlpha_epoch", format="pdf")
    plt.show()

    # Register the specific Times New Roman font
    times_new_roman_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    times_new_roman_font = font_manager.FontProperties(fname=times_new_roman_path)
    # Create DataFrame (assuming data[0] is available)
    df = pd.DataFrame(data[0])
    # Extract the series
    average_test_returns = df.get("LossQ")
    # Generate epoch indices
    epochs = range(1, len(average_test_returns) + 1)
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, average_test_returns, color="blue")
    plt.xlabel("Number of Epochs", fontproperties=times_new_roman_font, fontsize=14)
    plt.ylabel("LossQ", fontproperties=times_new_roman_font, fontsize=14)
    # plt.title("Average Test Return vs Epochs", fontsize=14, fontproperties=times_new_roman_font)
    plt.xticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.yticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.xlim([0,len(average_test_returns)])
    plt.grid(True)
    plt.legend(prop=times_new_roman_font, fontsize=14)
    plt.tight_layout()
    plt.savefig(all_logdirs[-1] + "/LossQ_epoch", format="pdf")
    plt.show()

    # Register the specific Times New Roman font
    times_new_roman_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    times_new_roman_font = font_manager.FontProperties(fname=times_new_roman_path)
    # Create DataFrame (assuming data[0] is available)
    df = pd.DataFrame(data[0])
    # Extract the series
    average_test_returns = df.get("AverageQ1Vals")
    # Generate epoch indices
    epochs = range(1, len(average_test_returns) + 1)
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, average_test_returns, color="blue")
    plt.xlabel("Number of Epochs", fontproperties=times_new_roman_font, fontsize=14)
    plt.ylabel("AverageQ1Vals", fontproperties=times_new_roman_font, fontsize=14)
    # plt.title("Average Test Return vs Epochs", fontsize=14, fontproperties=times_new_roman_font)
    plt.xticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.yticks(fontproperties=times_new_roman_font, fontsize=12)
    plt.xlim([0,len(average_test_returns)])
    plt.grid(True)
    plt.legend(prop=times_new_roman_font, fontsize=14)
    plt.tight_layout()
    plt.savefig(all_logdirs[-1] + "/AverageQ1Vals_epoch", format="pdf")
    plt.show()
    print("")

def make_plots_alpha(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean'):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value="Alpha", condition=condition, smooth=smooth, estimator=estimator,zorder=1)
    # plt.hlines(44.47124, 0, 10e4, 'k', linestyles="dashed", label='PID only')
    # plt.hlines( 68.5, 0, 10e5, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 91, 0, 1.36e6, 'k', linestyles="dashed", label='PI only')
    plt.legend()
    # plt.ylim([0,3])
    plt.savefig(all_logdirs[-1]+"/Alpha", format="png", bbox_inches='tight',zorder=2)
    plt.show()

    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value="AverageLogPi", condition=condition, smooth=smooth, estimator=estimator,zorder=1)
    # plt.hlines(44.47124, 0, 10e4, 'k', linestyles="dashed", label='PID only')
    # plt.hlines( 68.5, 0, 10e5, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 91, 0, 1.36e6, 'k', linestyles="dashed", label='PI only')
    plt.legend()
    # plt.ylim([-5,10])
    plt.savefig(all_logdirs[-1]+"/AverageLogPi", format="png", bbox_inches='tight',zorder=2)
    plt.show()

    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value="AverageTestEpRet", condition=condition, smooth=smooth, estimator=estimator,zorder=1)
    # plt.hlines(44.47124, 0, 10e4, 'k', linestyles="dashed", label='PID only')
    # plt.hlines( 68.5, 0, 10e5, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 91, 0, 1.36e6, 'k', linestyles="dashed", label='PI only')
    plt.hlines( 66.03, 0, 1.632e6, 'k', linestyles="dashed", label='PI only')
    plt.legend()
    # plt.ylim([-5,10])
    plt.savefig(all_logdirs[-1]+"/AverageTestEpRet", format="png", bbox_inches='tight',zorder=2)
    plt.show()

    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value="AverageEpRet", condition=condition, smooth=smooth, estimator=estimator,zorder=1)
    # plt.hlines(44.47124, 0, 10e4, 'k', linestyles="dashed", label='PID only')
    # plt.hlines( 68.5, 0, 10e5, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 91, 0, 1.36e6, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 107, 0, 1.632e6, 'k', linestyles="dashed", label='PI only')
    plt.hlines( 66.03, 0, 1.632e6, 'k', linestyles="dashed", label='PI only')
    plt.legend()
    # plt.ylim([-5,10])
    plt.savefig(all_logdirs[-1]+"/AverageEpRet", format="png", bbox_inches='tight',zorder=2)
    plt.show()


    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value="LossAlpha", condition=condition, smooth=smooth, estimator=estimator,zorder=1)
    # plt.hlines(44.47124, 0, 10e4, 'k', linestyles="dashed", label='PID only')
    # plt.hlines( 68.5, 0, 10e5, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 91, 0, 1.36e6, 'k', linestyles="dashed", label='PI only')
    plt.legend()
    # plt.ylim([-10,6])
    plt.savefig(all_logdirs[-1]+"/LossAlpha", format="png", bbox_inches='tight',zorder=2)
    plt.show()

    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value="LossQ", condition=condition, smooth=smooth, estimator=estimator,zorder=1)
    # plt.hlines(44.47124, 0, 10e4, 'k', linestyles="dashed", label='PID only')
    # plt.hlines( 68.5, 0, 10e5, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 91, 0, 1.36e6, 'k', linestyles="dashed", label='PI only')
    plt.legend()
    # plt.ylim([0,50])
    plt.savefig(all_logdirs[-1]+"/LossQ", format="png", bbox_inches='tight',zorder=2)
    plt.show()

    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value="AverageQ1Vals", condition=condition, smooth=smooth, estimator=estimator,zorder=1)
    # plt.hlines(44.47124, 0, 10e4, 'k', linestyles="dashed", label='PID only')
    # plt.hlines( 68.5, 0, 10e5, 'k', linestyles="dashed", label='PI only')
    # plt.hlines( 91, 0, 1.36e6, 'k', linestyles="dashed", label='PI only')
    plt.legend()
    # plt.ylim([0,40])
    plt.savefig(all_logdirs[-1]+"/AverageQ1Vals", format="png", bbox_inches='tight',zorder=2)
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """
    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count, 
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est)

    # make_plots_alpha(args.logdir, args.legend, args.xaxis, args.value, args.count,
    #            smooth=args.smooth, select=args.select, exclude=args.exclude,
    #            estimator=args.est)

if __name__ == "__main__":
    # exp_name = "Tworrv0_1"
    # logdir_manual = '/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Tworrv0_1'+exp_name
    main()