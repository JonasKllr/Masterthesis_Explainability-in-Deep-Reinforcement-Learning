import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lineartree import LinearTreeRegressor, LinearBoostRegressor


from read_buffer_from_file import read_buffer_from_file
DECISION_TREE = True
LINEAR_TREE = False
LINEAR_BOOST_TREE = False
LINEAR_REGRESSION = False

BUFFER_DIR = "/media/jonas/SSD_new/CMS/Semester_5/Masterarbeit/plots/final/explanations_4mil_final/plots/explainer_1_interrupted/2024-08-15_22-00/buffer"

states, actions = read_buffer_from_file(BUFFER_DIR)
np.random.seed(42)
np.random.shuffle(states)
np.random.shuffle(actions)

random_sample_id = np.random.choice(
    states.shape[0], size=100000, replace=False
)
states = states[random_sample_id]
actions = actions[random_sample_id]

train_size = int(0.8 * np.shape(states)[0])
train_states = states[:train_size]
train_actions = actions[:train_size]
test_states = states[train_size:]
test_actions = actions[train_size:]

feature_names = [
    "feature_0",
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
]

plt.hist(actions, bins=50, density=False)
plt.xlabel("agent action [-]")
plt.ylabel("Frequency [-]")
plt.show()
plt.close("all")

plt.hist(states[:,0], bins=50, density=False)
plt.xlabel("y-acceleration agent [m/s²]")
plt.ylabel("Frequency [-]")
plt.show()
plt.close("all")

plt.hist(states[:,1], bins=50, density=False)
plt.xlabel("y-speed agent [-]")
plt.ylabel("Frequency [-]")
plt.show()
plt.close("all")

plt.hist(states[:,2], bins=50, density=False)
plt.xlabel("x-distance agent-obstacle [m]")
plt.ylabel("Frequency [-]")
plt.show()
plt.close("all")

plt.hist(states[:,3], bins=50, density=False)
plt.xlabel("y-distance agent-obstacle [m]")
plt.ylabel("Frequency [-]")
plt.show()
plt.close("all")

plt.hist(states[:,4], bins=50, density=False)
plt.xlabel("y-speed difference agent-obstacle [m/s]")
plt.ylabel("Frequency [-]")
plt.show()
plt.close("all")

plt.hist(states[:,5], bins=50, density=False)
plt.xlabel("x-speed difference agent-obstacle [m/s]")
plt.ylabel("Frequency [-]")
plt.show()
plt.close("all")




if DECISION_TREE:

    train_mse = np.zeros(7)
    train_mae = np.zeros(7)
    test_mse = np.zeros(7)
    test_mae = np.zeros(7)

    for i in [3, 4, 5, 6]:

        if not os.path.exists(os.path.join(BUFFER_DIR, "trees", f"depth_{i}")):
            os.mkdir(os.path.join(BUFFER_DIR, "trees", f"depth_{i}"))

        print("fitting linear model tree")
        surrogate_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=i)
        surrogate_tree.fit(train_states, train_actions)
        print(i)
        print(surrogate_tree.get_depth())

        predictions_train = surrogate_tree.predict(train_states)
        predictions_test = surrogate_tree.predict(test_states)

        print("train data")
        train_mse[i] = mean_squared_error(train_actions, predictions_train)
        train_mae[i] = mean_absolute_error(train_actions, predictions_train)
        # print(f"mse: {mse}")
        # print(f"mae: {mae}")

        print("test data")
        test_mse[i] = mean_squared_error(test_actions, predictions_test)
        test_mae[i] = mean_absolute_error(test_actions, predictions_test)
        # print(f"mse: {mse}")
        # print(f"mae: {mae}")



        # plt.figure(figsize=(25, 10))
        # plot_tree(
        #     decision_tree=surrogate_tree,
        #     feature_names=feature_names,
        #     max_depth=None,
        #     filled=True,
        #     fontsize=12,
        #     proportion=True,
        # )
        # # plt.savefig(os.path.join(PLOT_DIR_TREE, f"{total_steps}_tree.pdf"))
        # plt.show()
        # plt.clf()
        # plt.close("all")


        # Save the image to a file
        # image = surrogate_tree.plot_model()
        # with open(os.path.join(BUFFER_DIR, 'output_image.png'), 'xb') as f:
        #     f.write(image.data)

        with open(os.path.join(BUFFER_DIR, "trees", f"depth_{i}", 'surrogate_tree_model.pkl'), 'xb') as file:
            pickle.dump(surrogate_tree, file)
    
    with open(os.path.join(BUFFER_DIR, "trees", 'train_mse.csv'), 'xb') as file:
        np.savetxt(file, train_mse, fmt='%1.5f')
    with open(os.path.join(BUFFER_DIR, "trees", 'train_mae.csv'), 'xb') as file:
        np.savetxt(file, train_mae, fmt='%1.5f')
    with open(os.path.join(BUFFER_DIR, "trees", 'test_mse.csv'), 'xb') as file:
        np.savetxt(file, test_mse, fmt='%1.5f')
    with open(os.path.join(BUFFER_DIR, "trees", 'test_mae.csv'), 'xb') as file:
        np.savetxt(file, test_mae, fmt='%1.5f')


        PRUNING = False
        if PRUNING:
            path = surrogate_tree.cost_complexity_pruning_path(train_states, train_actions)
            ccp_alphas = path.ccp_alphas
            impurities = path.impurities

            # random_sample_id = np.random.choice(
            #     ccp_alphas.shape[0], size=50, replace=False
            # )
            ccp_alphas = ccp_alphas[::100]
            impurities = impurities[::100]


            # Plot the total impurity vs effective alpha for visualization
            plt.figure(figsize=(8, 6))
            plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
            plt.xlabel("Effective Alpha")
            plt.ylabel("Total Impurity of Leaves")
            plt.title("Total Impurity vs Effective Alpha")
            plt.show()

            ccp_alphas = np.array([3*10**(-5), 4*10**(-5), 4.2*10**(-5), 4.6*10**(-5)])

            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = DecisionTreeRegressor(random_state=0, max_depth=40, ccp_alpha=ccp_alpha)
                clf.fit(train_states, train_actions)
                clfs.append(clf)
            
            # clfs = clfs[:-1]
            # ccp_alphas = ccp_alphas[:-1]

            node_counts = [clf.tree_.node_count for clf in clfs]
            depth = [clf.tree_.max_depth for clf in clfs]
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
            ax[0].set_xlabel("alpha")
            ax[0].set_ylabel("number of nodes")
            ax[0].set_title("Number of nodes vs alpha")
            ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
            ax[1].set_xlabel("alpha")
            ax[1].set_ylabel("depth of tree")
            ax[1].set_title("Depth vs alpha")
            fig.tight_layout()
            plt.show()

            train_scores = [clf.score(train_states, train_actions) for clf in clfs]
            test_scores = [clf.score(test_states, test_actions) for clf in clfs]

            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("accuracy")
            ax.set_title("Accuracy vs alpha for training and testing sets")
            ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
            ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
            ax.legend()
            plt.show()

elif LINEAR_TREE:
    # train_states[:,3] *= 0.7
    # train_states[:,4] *= 0.9

    surrogate_tree = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        criterion="mse",
        max_depth = 15,
        min_samples_leaf=0.0025,
        max_bins=100,
        # split_features=[3],
        #linear_features=[3, 4]
    )
    print("fitting linear model tree")
    surrogate_tree.fit(train_states, train_actions)
    predictions_train = surrogate_tree.predict(train_states)
    predictions_test = surrogate_tree.predict(test_states)

    print("train data")
    r_squared = r2_score(train_actions, predictions_train)
    print(f"r_squared:{r_squared}")
    mse = mean_squared_error(train_actions, predictions_train)
    print(f"mse: {mse}")
    mae = mean_absolute_error(train_actions, predictions_train)
    print(f"mae: {mae}")

    print("test data")
    r_squared = r2_score(test_actions, predictions_test)
    print(f"r_squared:{r_squared}")
    mse = mean_squared_error(test_actions, predictions_test)
    print(f"mse: {mse}")
    mae = mean_absolute_error(test_actions, predictions_test)
    print(f"mae: {mae}")




    # Save the image to a file
    image = surrogate_tree.plot_model()
    with open(os.path.join(BUFFER_DIR, 'output_image.png'), 'xb') as f:
        f.write(image.data)

    with open(os.path.join(BUFFER_DIR, 'linear_tree_model.pkl'), 'xb') as file:
        pickle.dump(surrogate_tree, file)

elif LINEAR_BOOST_TREE:
    surrogate_tree = LinearBoostRegressor(
        base_estimator = LinearRegression(),
        max_depth = 7,
        min_samples_leaf=0.0025,
        # split_features=[3],
        #linear_features=[3, 4]
    )
    print("fitting linear model tree")
    surrogate_tree.fit(train_states, train_actions)
    predictions_train = surrogate_tree.predict(train_states)
    predictions_test = surrogate_tree.predict(test_states)

    print("train data")
    r_squared = r2_score(train_actions, predictions_train)
    print(f"r_squared:{r_squared}")
    mse = mean_squared_error(train_actions, predictions_train)
    print(f"mse: {mse}")
    mae = mean_absolute_error(train_actions, predictions_train)
    print(f"mae: {mae}")

    print("test data")
    r_squared = r2_score(test_actions, predictions_test)
    print(f"r_squared:{r_squared}")
    mse = mean_squared_error(test_actions, predictions_test)
    print(f"mse: {mse}")
    mae = mean_absolute_error(test_actions, predictions_test)
    print(f"mae: {mae}")




    # Save the image to a file
    # image = surrogate_tree.plot_model()
    # with open(os.path.join(BUFFER_DIR, 'output_image.png'), 'xb') as f:
    #     f.write(image.data)

    # plot_tree(
    #     decision_tree=surrogate_tree,
    #     feature_names=feature_names,
    #     max_depth=None,
    #     filled=True,
    #     fontsize=12,
    #     proportion=True,
    # )

    with open(os.path.join(BUFFER_DIR, 'linear_tree_model.pkl'), 'xb') as file:
        pickle.dump(surrogate_tree, file)
    

    
elif LINEAR_REGRESSION:
    train_states[:,3] *= 5
    train_states[:,4] *= 4
    train_states[:,0] = 0
    train_states[:,1] = 0
    train_states[:,2] = 0
    train_states[:,5] = 0
    regressor = LinearRegression().fit(X=train_states, y=train_actions)
    # regressor = Lasso(alpha=1).fit(X=train_states, y=train_actions)
    print(regressor.coef_)

    predictions_train = regressor.predict(train_states)
    predictions_test = regressor.predict(test_states)

    print("train data")
    r_squared = r2_score(train_actions, predictions_train)
    print(f"r_squared:{r_squared}")
    mse = mean_squared_error(train_actions, predictions_train)
    print(f"mse: {mse}")
    mae = mean_absolute_error(train_actions, predictions_train)
    print(f"mae: {mae}")

    print("test data")
    r_squared = r2_score(test_actions, predictions_test)
    print(f"r_squared:{r_squared}")
    mse = mean_squared_error(test_actions, predictions_test)
    print(f"mse: {mse}")
    mae = mean_absolute_error(test_actions, predictions_test)
    print(f"mae: {mae}")

    with open(os.path.join(BUFFER_DIR, 'linear_regression.pkl'), 'xb') as file:
        pickle.dump(regressor, file)


