#
#   Decision tree algorithm using pandas and sklearn (scikit-learn)
#  
#   Creates a Decision Tree based on the contents of "data.csv" (or otherwise specified), 
#   and saves a visualization of the tree in "tree.png" (or otherwise specified).
#   If specified, runs tests and save results to "results.csv" (or otherwise specified).
#       
#
# TODO: maybe rename the command line arguments?

import os
import time
import argparse
import pandas
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz

DEPTH_ABSOLUTE_MAXIMUM = 100    # Change this for your limit when running bestDepth() or minDepth()

# Additional functions for choosing tree depth
# Tries every depth from i to DEPTH_ABSOLUTE_MAXIMUM and returns the tree with best accuracy
def bestDepth(X_train, X_test, y_train, y_test):
    best_acc = 0.0
    for i in range(1, DEPTH_ABSOLUTE_MAXIMUM):
        dtree = DecisionTreeClassifier(max_depth=i)
        dtree = dtree.fit(X_train, y_train)
        y_pred = dtree.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        # print(f"[{i}]: {acc}")

        if (acc > best_acc):
            best_acc = acc
            best_depth = i
            best_tree = dtree

    print(f"Best depth: {best_depth}, with Accuracy of: {best_acc:.4f}.")
    return best_tree

# Tries every depth from DEPTH_ABSOLUTE_MAXIMUM to i and returns the last tree with > 0.85 (defined in argparse in main())
def minDepth(X_train, X_test, y_train, y_test, args):
    print(f"Using minimum accuracy of: {args.accuracy}")
    best_tree = None
    best_depth = 0
    best_acc = 0.0
    
    for current_depth in range(1, DEPTH_ABSOLUTE_MAXIMUM):
        dtree = DecisionTreeClassifier(max_depth=current_depth)
        dtree = dtree.fit(X_train, y_train)
        y_pred = dtree.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        # print(f"[{i}]: {acc}")

        # If got to the target accuracy, saves and ends
        if acc >= args.accuracy:
            best_tree = dtree
            best_depth = current_depth
            best_acc = acc
            break
        # Else, keep running 
        elif acc > best_acc or best_tree is None:
            best_tree = dtree
            best_depth = current_depth
            best_acc = acc
    
    # After checking all depths or finding the first suitable one
    if best_acc >= args.accuracy:
        print(f"Minimum depth: {best_depth}, with Accuracy of : {best_acc:.4f};")
    else:
        print(f"No tree met accuracy requirement. Returning best found. Depth: {best_depth}, with Accuracy of {best_acc:.4f}.")
    
    return best_tree

def drawGraphGraphviz(dtree, attributes, args):
    dot_data = tree.export_graphviz(
                                dtree, out_file=None, feature_names=attributes,
                                filled=True
                                )

    # Change this to be the one specified by the user:
    filename, extension = os.path.splitext(args.output)
    extension = extension[1:]
    # Render
    graph = graphviz.Source(dot_data, format=extension)
    graph.render(filename)
    os.remove(filename)

def drawGraphMatplot(dtree, attributes, args):
    tree.plot_tree(
                dtree, feature_names=attributes, 
                filled=True
                )
    # Render
    plt.savefig(args.output, dpi=700)

def saveResults(results, args):
    # Calculate averages
    averages = results.mean(numeric_only=True).to_frame().T 
    averages['Run'] = 'Average'

    # Add to the final file, save as .csv
    final_results = pandas.concat([averages, results], ignore_index=True)
    final_results.to_csv(args.metric, index=False)
    
    # Summary just because
    print("\nBenchmark Summary:")
    print(f"Average training time: {averages['Training Time'].values[0]:.4f} seconds")
    print(f"Average testing time: {averages['Testing Time'].values[0]:.4f} seconds")
    print(f"Average accuracy: {averages['Accuracy'].values[0]:.4f}")
    print(f"Average F-measure: {averages['F-measure'].values[0]:.4f}")

def runBenchmark(X, y, args):
    # Our ""defines""
    MAX_RUN = 1000          # How many runs to do
    MAGIC_NUMBER = 39       # Doesn't really matter. More for flare, to be honest
    
    results = []  
    
    # Do runs
    print(f"Executing {MAX_RUN} runs:")
    for i in range(MAX_RUN):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, random_state=MAGIC_NUMBER+i)
        
        dtree = DecisionTreeClassifier(random_state=MAGIC_NUMBER+i)

        # Benchmark training
        start_training = time.time()
        dtree.fit(X_train, y_train)
        training_time = time.time() - start_training

        # Benchmark testing
        start_testing = time.time()
        y_pred = dtree.predict(X_test)
        testing_time = time.time() - start_testing

        # Metrics
        acc = metrics.accuracy_score(y_test, y_pred)
        f_measure = metrics.f1_score(y_test, y_pred, average='weighted')

        # Save results
        results.append({
            'Run': i+1,
            'Training Time': training_time,
            'Testing Time': testing_time,
            'Accuracy': acc,
            'F-measure': f_measure,
        })

    # Send to final processing
    saveResults(pandas.DataFrame(results), args)






def main():
    parser = argparse.ArgumentParser(
        description="Generates a Decision Tree for a given dataset. Saves metrics and a visualization of the working tree.",
        epilog='Example: python decisiontree.py --depth best --test 0.4 --file "data processed.csv" --output output.pdf --benchmark --metric results.csv'
        )

    parser.add_argument('-d', '--depth',
                        choices=['best', 'minimum', 'limited'],
                        help='Depth selection: "best" for the best accuracy found, "minimum" for the lowest with accuracy still greater than 0.85 or that set by --accuracy, "limited" for a limited depth level specified by --limit. Default renders to the maximum depth.'
                        )
    parser.add_argument('-a', '--accuracy',
                        type=float,
                        help='Minimum accuracy desired when running "--depth minimum". Default is 0.85',
                        default=0.85
                        )
    parser.add_argument('-l', '--limit',
                        type=int,
                        help='Maximum depth level when running "--depth limited". Default is 15',
                        default=15
                        )
    parser.add_argument('-t', '--test',
                        type=float,
                        help='Test subset ratio size. Default is 0.3 for 30%% test, 70%% train.',
                        default=0.3
                        )
    parser.add_argument('-r', '--render',
                        choices=['viz', 'plt'],
                        help='Which tool to use to render the tree image. "viz" for Graphviz (default), "plt" for Matplotlib.',
                        default='viz'
                        )
    parser.add_argument('-f', '--file',
                        help='Path to dataset file. Must be .csv',
                        default='data.csv'
                        )
    parser.add_argument('-o', '--output',
                        help='Path to the output image file. Default is \'tree.png\'. Can also be other extensions, such as png and jpeg.',
                        default='tree.png'
                        )
    parser.add_argument('-b', '--benchmark',
                        action='store_true',
                        help='Ignores depth specifications and runs code multiple times to find metrics. Save them to the file specified in -m.'
                        )
    parser.add_argument('-m', '--metric',
                        help='Path to the metrics output file. Default is \'metrics.csv\'.',
                        default='metrics.csv'
                        )

    args = parser.parse_args()


    # Read csv
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist")
        exit(1)
    df = pandas.read_csv(args.file)

    # Change from string to numerical
    d = {'Hazardous': 0, 'Poor': 1, 'Moderate': 2, 'Good': 3}
    df['Air Quality'] = df['Air Quality'].map(d)
    attributes =['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']

    # X are the attributes we use to predict Y
    X = df[attributes].values
    y = df['Air Quality'].values

    if (args.benchmark):
        runBenchmark(X, y, args)
        exit(0)

    # Divide into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, random_state=1)


    # Create tree image and save it
    match args.depth:
        case "best":
            print("Mode: Best")
            dtree = bestDepth(X_train, X_test, y_train, y_test)
        case "minimum":
            print("Mode: Minimum")
            dtree = minDepth(X_train, X_test, y_train, y_test, args)
        case "limited":
            print("Mode: Limited")
            print(f"Using maximum depth of: {args.limit}")
            dtree = DecisionTreeClassifier(max_depth=args.limit)
            dtree = dtree.fit(X_train, y_train)
            y_pred = dtree.predict(X_test)
            print(f"Depth: {args.limit}, with Accuracy of: {metrics.accuracy_score(y_test, y_pred):.4f}.")
        case _:
            print("Mode: Default")
            dtree = DecisionTreeClassifier()
            dtree = dtree.fit(X_train, y_train)
            y_pred = dtree.predict(X_test)
            print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
            
    # saves image to the output file
    match args.render:
        case "viz":
            print("Render: Graphviz")
            drawGraphGraphviz(dtree, attributes, args)
        case "plt":
            print("Render: Matplotlib")
            drawGraphMatplot(dtree, attributes, args)
        case _:
            print(f"Unexpected render value: {args.render}. Rendering with Graphviz.")
            drawGraphGraphviz(dtree, attributes, args)


            
    

if __name__ == '__main__':
    main()