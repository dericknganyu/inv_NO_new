import os
import argparse

def DeleteIfExist(dir, txt):
    files = os.listdir(dir)
    for fichier in files:
        if txt in fichier:
            os.remove(dir +'/'+ fichier)


        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='For inputing resolution')

    #Adding required parser argument
    parser.add_argument('--dir', default='/home/derick/Documents/inv_NO/figures/darcyPWC', type=str, help='Specify redolution')
    parser.add_argument('--txt', default='Results_100_samples_20220831-', type=str, help='Specify redolution')

    #parsing
    args = parser.parse_args()
    dir = args.dir
    txt = args.txt

    #DeleteIfExist(dir, txt)

    def DeleteFile(fichier, txt= txt ):
        if txt in fichier:
            print (fichier)
            os.remove(fichier)

    class DirWalker(object):

        def walk(self,dir,meth):
            dir = os.path.abspath(dir)
            for file in [file for file in os.listdir(dir) if not file in [".",".."]]:
                nfile = os.path.join(dir,file)
                meth(nfile)
                if os.path.isdir(nfile):
                    self.walk(nfile,meth)

    
    DirWalker().walk(dir= dir, meth = DeleteFile)