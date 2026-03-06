"""Utility to verify MongoDB connectivity using the project's configuration class."""

import os
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import MONGODB_URL_KEY


def main():
    url = os.getenv(MONGODB_URL_KEY)
    if url is None:
        print(f"Environment variable {MONGODB_URL_KEY} is not set.")
        return
    try:
        print(f"Attempting connection to {url.split('@')[-1]} ...")
        MongoDBClient.test_connection()
        print("Connection successful.")
    except Exception as e:
        print("Connection failed:", str(e))


if __name__ == "__main__":
    main()
