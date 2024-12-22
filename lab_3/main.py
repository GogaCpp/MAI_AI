import argparse
from plotly.io import show
import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
from sklearn.metrics import accuracy_score

# ! вроде как аргпарсер лишний, но для тестового запуска я не хочу ждать 5 минут так что так
parser = argparse.ArgumentParser()
parser.add_argument('-no_learn', action='store_false', help='if not need learn')
args = parser.parse_args()

storage_url = "postgresql://postgres:postgres@localhost:5432/postgres"
# даже не знал что бывает рак молочной железы
data = sklearn.datasets.load_breast_cancer()

X = data.data
y = data.target
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):  # Тут я определяю целевую функцию
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)

    clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# два принтера и два сэмплера. принтеры используются для ранней остановки проб, а сэмплеры — для выбора значений гиперпараметров.
pruners = {'MedianPruner': optuna.pruners.MedianPruner(), 'HyperbandPruner': optuna.pruners.HyperbandPruner()}
samplers = {'RandomSampler': optuna.samplers.RandomSampler(), 'TPESampler': optuna.samplers.TPESampler()}


if args.no_learn:
    #  запуск оптимизации по каждой комбинации принтера и сэмплера
    for pruner in pruners:
        for sampler in samplers:
            study_name = f'study_{pruner}_{sampler}'
            study = optuna.create_study(direction='maximize', pruner=pruners[pruner], sampler=samplers[sampler], study_name=study_name, storage=storage_url, load_if_exists=True)
            study.optimize(objective, n_trials=50)

# загрузка и сохранение результатов
# кайф что load_study загружает ранее сохраненное исследование по имени
studies = []
for pruner in pruners:
    for sampler in samplers:
        study_name = f'study_{pruner}_{sampler}' 
        saved_study = optuna.load_study(study_name=study_name, storage=storage_url)
        studies.append(saved_study)


def show_params(trial):
    print("Параметры: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


print("Лучшие результаты:")
for study in studies:
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    show_params(trial)

# создается график истории оптимизации
# ? мб уюрать show_params
for study in studies:
    trial = study.best_trial
    #show_params(trial)
    fig = optuna.visualization.plot_optimization_history(study)
    show(fig)

# создается график важности параметров
# ? мб убрать show_params
for study in studies:
    trial = study.best_trial
    #show_params(trial)
    fig = optuna.visualization.plot_param_importances(study)
    show(fig)