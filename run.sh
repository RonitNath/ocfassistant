#! /bin/bash

poetry run python main.py --scrape --depth 2 --extend-depth
poetry run python main.py --chunk
