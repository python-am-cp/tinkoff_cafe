import argparse
import sys
import importlib.util
import pandas as pd


def parse():
    parser = argparse.ArgumentParser(description='Прогнозрование.')
    parser.add_argument('--test', metavar='--in', type=str, default='test.csv',
                        help='название файла в формате test.csv')
    parser.add_argument('--menu', type=str, default='menu_train.csv', help='название файла с меню по дням')
    parser.add_argument('--tags', type=str, default='menu_tagged.csv',
                        help='название файла со всеми блюдами и их характеристиками')
    parser.add_argument('--model', type=str, default='solution.py', help='название файла c моделью')
    parser.add_argument('--prediction', type=str, default='prediction.csv',
                        help='название файла результатом предсказания')
    args = parser.parse_args(sys.argv[1:])
    return args.test, args.menu, args.tags, args.model, args.prediction


def main():
    test_fn, menu_test_fn, goods_tags_fn, model_fn, prediction_fn = parse()

    # Import model from path
    spec = importlib.util.spec_from_file_location("solution", model_fn)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    model = solution.Model()
    model.load_params(menu_test_fn, goods_tags_fn)
    test_df = pd.read_csv(test_fn)
    res = []
    for row in test_df.values:
        chknum, person_id, month, day, *_ = row
        labels = list(model.predict([person_id, day, month]))
        res.append([chknum, " ".join([str(i) for i in sorted(labels)])])
    prediction = pd.DataFrame(res, columns=['chknum', 'pred'])
    prediction.to_csv(prediction_fn, index=False)


if __name__ == '__main__':
    main()
