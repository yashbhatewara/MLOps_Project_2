import os
import sys
import pymongo
import certifi
import ssl
from pymongo.errors import PyMongoError
from src.exception import MyException
from src.constants import MONGODB_URL_KEY, DATABASE_NAME
from src.logger import logging

ca = certifi.where()


# helper for environment details (useful for debugging handshake errors)
def _log_ssl_info():
    try:
        logging.info(f"Python version: {sys.version.split()[0]}")
        logging.info(f"OpenSSL version: {ssl.OPENSSL_VERSION}")
        logging.info(f"pymongo version: {pymongo.__version__}")
        logging.info(f"certifi CA bundle: {ca}")
    except Exception:
        pass


class MongoDBClient:
    """
    Connects to the MongoDB database using the URL from environment variables.

    The class caches a single client instance so that subsequent constructors
    reuse the same connection pool.  Optional TLS parameters may be driven
    from environment variables to assist debugging connection issues.
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable: {MONGODB_URL_KEY} is not set.")

                # allow overriding tls options via environment for diagnostics
                tls = os.getenv("MONGODB_TLS", "true").lower() in ("1", "true", "yes")
                tls_allow_invalid = os.getenv("MONGODB_TLS_ALLOW_INVALID_CERTS", "false").lower() in ("1", "true", "yes")

                logging.info(f"Attempting MongoDB connection to {mongo_db_url.split('@')[-1]}")
                _log_ssl_info()
                try:
                    MongoDBClient.client = pymongo.MongoClient(
                        mongo_db_url,
                        tls=tls,
                        tlsCAFile=ca,
                        tlsAllowInvalidCertificates=tls_allow_invalid,
                    )
                    # force a server call so errors surface early
                    MongoDBClient.client.server_info()
                except (PyMongoError, ssl.SSLError) as exc:
                    # log diagnostic info for the error
                    logging.error("Exception during MongoDB connection attempt:", exc_info=True)
                    msg = (
                        "failed to establish SSL/TLS connection to MongoDB. "
                        "This is often caused by an incompatible OpenSSL/Python "
                        "version, missing CA certificates, or a network/firewall "
                        "blocking the handshake. Make sure your Atlas IP whitelist "
                        "allows your current address, and that you can reach the "
                        "cluster using `openssl s_client -connect <host>:27017`.")
                    logging.error(msg)
                    raise

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB client initialized successfully")
        except Exception as e:
            raise MyException(e, sys) from e

    @classmethod
    def test_connection(cls) -> bool:
        """Attempts a lightweight call to the server to verify connectivity.

        Returns
        -------
        bool
            True if the connection succeeds, otherwise an exception is raised.

        This is useful for health checks or CLI scripts. It does not cache a
        client instance when called separately.
        """
        mongo_db_url = os.getenv(MONGODB_URL_KEY)
        if mongo_db_url is None:
            raise Exception(f"Environment variable: {MONGODB_URL_KEY} is not set.")

        tls = os.getenv("MONGODB_TLS", "true").lower() in ("1", "true", "yes")
        tls_allow_invalid = os.getenv("MONGODB_TLS_ALLOW_INVALID_CERTS", "false").lower() in ("1", "true", "yes")

        _log_ssl_info()
        temp_client = pymongo.MongoClient(
            mongo_db_url,
            tls=tls,
            tlsCAFile=ca,
            tlsAllowInvalidCertificates=tls_allow_invalid,
        )
        # this call will raise if the server cannot be reached
        temp_client.server_info()
        logging.info("MongoDB test connection successful")
        return True
