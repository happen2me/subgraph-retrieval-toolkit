import socket
from urllib.parse import urlparse


def get_host_port(url):
    """Get the host and port from a URL"""
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port
    if port is None:
        if parsed_url.scheme == 'http':
            port = 80
        elif parsed_url.scheme == 'https':
            port = 443
    return host, port


def socket_reachable(url):
    """Check if a socket is reachable
    """
    host, port = get_host_port(url)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2) # set a timeout value for the socket
        s.connect((host, port))
        s.close()
        return True
    except Exception as err:
        print(err)
        return False
