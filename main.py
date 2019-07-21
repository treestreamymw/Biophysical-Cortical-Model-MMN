from glob import glob
from figure_utils import create_plot, exctract_data
from variation_utils import set_var
from config import SAMPLE_VAR, N_PEAKS, N_MSRMNT, JSONS_DIR
import argparse

## plot
def get_plot():
    all_files_names = glob('{}/*.json'.format(JSONS_DIR))

    calc_data = exctract_data(all_files_names, N_PEAKS, N_MSRMNT)
    create_plot (calc_data)

## variation code
def get_var_from_file(file_path):
    # to be implemeted
    pass

def run_var(given_var=None):
    #file_name = 'Ca_HVA.mod'

    var = SAMPLE_VAR or given_var

    set_var(var)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", help='Plot LFP mean', action="store_true")
    parser.add_argument("--set_var", help='set variation', action="store_true")

    args = parser.parse_args()
    if args.plot:
        get_plot()
        return

    elif args.set_var:
        var1 = [{'gene_name':'CACNA1C',
                'ref':'27,S24',
                'channel':'CA',
                'variation':[{'offm': -25.9, 'offh': -27,'c' : 0.066}]}]

        var2 =[{'gene_name':'CACNA1C',
                'ref':'27,S24',
                'channel':'CA',
                'variation':[{'offm': -37.3, 'offh': -30,'c' : 0.042}]}]

        var3 = [{'gene_name':'CACNA1C',
                'ref':'31,S29',
                'channel':'CA',
                'variation':[{'offm': 1, 'offh': -3.1,'sloh':1.24,'c':0.236}]}]

        var4 = [{'gene_name':'CACNA1C',
                'ref':'32,33,S30,S31',
                'channel':'CA',
                'variation':[{'offm': -10.9, 'slom':0.73,'offh': -3,\
                        'sloh':0.81,'tauh':1.25,'c' : 0.083}]}]
        run_var(var1)

if __name__ == '__main__':
    main()
