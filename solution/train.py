
import argparse
import sys
import importlib.util


def parse():
    parser = argparse.ArgumentParser(description='Обучение модели.')
    parser.add_argument('--train', metavar='--in', type=str, default='train.csv',
                        help='название файла в формате train.csv')
    parser.add_argument('--menu', type=str, default='menu_train.csv', help='название файла с меню по дням')
    parser.add_argument('--tags', type=str, default='menu_tagged.csv',
                        help='название файла со всеми блюдами и их характеристиками')
    parser.add_argument('--model', type=str, default='solution.py', help='название файла c моделью')
    args = parser.parse_args(sys.argv[1:])
    return args.train, args.menu, args.tags, args.model


def main():
    train_fn, menu_train_fn, goods_tags_fn, model_fn = parse()

    # Import model from path
    spec = importlib.util.spec_from_file_location("solution", model_fn)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    model = solution.Model()
    model.train(train_fn, menu_train_fn, goods_tags_fn)


if __name__ == '__main__':
    main()
