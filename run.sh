#! /bin/bash

poetry run python main.py --scrape --depth 1
poetry run python main.py --chunk
