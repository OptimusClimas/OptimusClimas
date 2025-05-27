def findpossiblehyperparameters():
    # searches for possible hyperparameters for the mViT using a conditional equation(see Appendix B)
    # p patch size, h hiddensize, n number of filters, m number of conv, f filtersize
    x = 20
    y = 140
    p = []
    h = []
    mm = []
    ff = []
    nn = []
    for patch_size in range(25, 100):
        for hidden_size in range(200, 1000):
            for m in range(2, 8):
                for f in range(3, 10):
                    for n in range(10, 100):
                        if ((x // patch_size) * (y // patch_size) * hidden_size) == (
                                (x - ((m - 1) * (f - 1))) * (y - ((m - 1) * (f - 1))) * n):
                            p.append(patch_size)
                            h.append(hidden_size)
                            mm.append(m)
                            ff.append(f)
                            nn.append(n)
    return p, h, mm, ff, nn
