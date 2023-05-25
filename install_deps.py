from subprocess import call, Popen
import os

def main(force_reinstall=False):

    if not force_reinstall and os.path.exists('/usr/local/lib/python3.10/dist-packages/safetensors'):
        # ntbks()
        print('[1;32mModules and notebooks updated, dependencies already installed')

    else:
        print('[1;33mInstalling the dependencies...')
        call('pip install --root-user-action=ignore --disable-pip-version-check --no-deps -qq gdown numpy==1.23.5 accelerate==0.12.0 --force-reinstall', shell=True, stdout=open('/dev/null', 'w'))
        # ntbks()
        if os.path.exists('deps'):
            call("rm -r deps", shell=True)
        if os.path.exists('diffusers'):
            call("rm -r diffusers", shell=True)
        call('mkdir deps', shell=True)
        if not os.path.exists('cache'):
            call('mkdir cache', shell=True)
        os.chdir('deps')
        
        call('wget -q https://huggingface.co/flix-k/sd_dependencies/resolve/main/rnpddeps-t2.tar.zst', shell=True, stdout=open('/dev/null', 'w'))
        # call('dpkg -i *.deb', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
        call('tar -C / --zstd -xf rnpddeps-t2.tar.zst', shell=True, stdout=open('/dev/null', 'w'))
        # Popen('apt-get install libfontconfig1 libgles2-mesa-dev -q=2 --no-install-recommends', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
        call("sed -i 's@~/.cache@/workspace/cache@' /usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", shell=True)
        os.chdir('/workspace')
        call("git clone --depth 1 -q https://github.com/flixmk/diffusers", shell=True, stdout=open('/dev/null', 'w'))
        call("rm -r deps", shell=True)
        call("pip install datasets timm", shell=True)
        call("pip install git+https://github.com/flixmk/clean-fid.git", shell=True)
        call("pip install pytorch-lightning", shell=True)
        call("pip install lightning", shell=True)
        call("pip install h5py", shell=True)
        call("pip install -U jsonargparse[signatures]", shell=True)
        call("pip install scikit-learn", shell=True)
        # call("pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117", shell=True)

        
if __name__ == "__main__":
    main()

