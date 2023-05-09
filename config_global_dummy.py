# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

enb = None


def print_this(s, x):
    x = x.float()
    print(s,
          x.min().item(),
          x.mean().item(),
          x.max().item(),
          x.size()
          )
