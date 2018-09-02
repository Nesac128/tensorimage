def snd_itcl(idata):
    odata = []
    for n, sfd in enumerate(idata):
        sdt = []
        for iv in sfd:
            sdt += iv
        odata.append(sdt)
    return odata