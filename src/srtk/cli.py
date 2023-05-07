"""Command-line interface for SRTK."""
import argparse

from .link import _add_arguments as add_link_arguments
from .link import link
from .preprocess import _add_arguments as add_preprocess_arguments
from .preprocess import preprocess
from .retrieve import _add_arguments as add_retrieve_arguments
from .retrieve import retrieve
from .train import _add_arguments as add_train_arguments
from .train import train
from .visualize import _add_arguments as add_visualize_arguments
from .visualize import visualize


def main():
    """Main entry to the command line interface.
    """
    parser = argparse.ArgumentParser(description='SRTK: A toolkit for smantic-relevant subgraph retrieval')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_link = subparsers.add_parser('link', help='link entities to a knowledge graph')
    add_link_arguments(parser_link)
    parser_link.set_defaults(func=link)

    parser_preprocess = subparsers.add_parser('preprocess', help='preprocess the data')
    add_preprocess_arguments(parser_preprocess)
    parser_preprocess.set_defaults(func=preprocess)

    parser_train = subparsers.add_parser('train', help='train a subgraph retriever')
    add_train_arguments(parser_train)
    parser_train.set_defaults(func=train)

    parser_retrieve = subparsers.add_parser('retrieve', help='retrieve a subgraph with a natural query')
    add_retrieve_arguments(parser_retrieve)
    parser_retrieve.set_defaults(func=retrieve)

    parser_visualize = subparsers.add_parser('visualize', help='visualize the retrieved subgraph')
    add_visualize_arguments(parser_visualize)
    parser_visualize.set_defaults(func=visualize)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
