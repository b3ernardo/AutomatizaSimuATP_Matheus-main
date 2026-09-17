from subprocess import run
from time import sleep
from pathlib import Path

def runATP(path: str) -> None:

    run(['runATP.exe', path], shell=False, check=True, stdout=False)
    sleep(3)


if __name__ == '__main__':
    path = Path(__file__).resolve().parent / "step1_cartao/SistemaNovoMatheus1.atp"
    runATP(path)
