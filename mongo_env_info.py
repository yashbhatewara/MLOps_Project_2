"""Print Python, OpenSSL and pymongo information to help debug TLS errors."""

import ssl
import sys
import pymongo
import certifi


def main():
    print(f"Python: {sys.version}")
    print(f"OpenSSL: {ssl.OPENSSL_VERSION}")
    print(f"pymongo: {pymongo.__version__}")
    print(f"certifi CA bundle: {certifi.where()}")


if __name__ == "__main__":
    main()
