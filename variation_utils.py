
import re
import subprocess as sp
import multiprocessing as mp
from config import VAR_FACTOR_TYPE
from genetic_variations import SAMPLE_VAR, SINGLE_VAR, MULTI_GENE_VAR

from decimal import *
import shutil

## utils ##

def float_to_str(f):

    ctx = Context()
    ctx.prec =10

    d1 = ctx.create_decimal(repr(f))

    return format(d1, 'f')

def factor_calc (factor_type, current_val, variation_val, c):
    if factor_type == 'add':
        # existing value + new value * c
        return current_val + (variation_val*c)

    elif factor_type == 'mul':
        # existing value * new value ^ c
        return current_val * (variation_val**c)

def open_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return content

def update_file(filename, content):
    with open(filename, "w") as f:
        f.write(content)
    return True

def get_param_section_indexes(mod_content_list):
    # extracts parameter section
    # return indexes for the param section

    flag = False
    line_indexes = []

    for counter, line in enumerate(mod_content_list):

        if '}' in line:
            flag= False
        if flag :

            line_indexes.append(counter)

        if 'PARAMETER' in line:
            flag = True

    return line_indexes


def prepare_param_section_list(new_params, param_units ):
    # from dict of values to a well formatted list
    new_unit_param_section_list = []
    template = '        {} = {} ({})\n'

    for param_name, param_value in new_params.items():
        param_unit = param_units[param_name]
        new_unit_param_section_list.append(template.\
            format(param_name, float_to_str(param_value), param_unit))

    return new_unit_param_section_list


def calc_new_vals(variation_setting, curr_values, factor_rules = VAR_FACTOR_TYPE):
    # update params values

    # copy
    new_params = curr_values.copy()

    for variation in variation_setting:
        # get c
        c = variation['c']

        variation_to_modify = variation.keys()

        for param in curr_values:
            param_name = param[:-1]

            if param_name in variation_to_modify:
                factor_type = factor_rules.get(param_name)

                if factor_type:
                    new_val = factor_calc(factor_type, curr_values[param], \
                            variation[param_name],c)

                new_params[param]=new_val


    return new_params

def restore_default():
    to_path = 'mod_files'
    from_path = 'default_params_mod_files'

    shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def update_content(content, new_params_list, line_indexes):
    # insert new values to the full list from the original file
    print(new_params_list)
    print(line_indexes)
    assert (len(new_params_list)== len(line_indexes))

    for line_ind, line_content in zip(line_indexes, new_params_list):
        content[line_ind]= line_content

    return ''.join(content)


def get_def_info(mod_content_list, indexes):
    param_vals = {} #name:value
    param_units= {} #name:unit

    data = [mod_content_list[i] for i in indexes]

    for line in data:
        txt = line.strip()

        # find param name
        param_name_ind=txt.find('\t' or '=') \
                if txt.find('\t' or '=') !=-1 else txt.find('=')
        param_name=txt[:param_name_ind].strip()

        #find param value
        param_name_ind=txt.find('=')
        param_value_ind=txt.find('\t' or '(', param_name_ind) \
            if txt.find('\t' or '(', param_name_ind)!=-1 \
            else txt.find('(', param_name_ind)

        param_value=txt[param_name_ind+1:param_value_ind].strip()
        try:
            param_vals[param_name]=float(param_value)
        except ValueError:
            print("extracting values of file err. value is not a float")



        #find param unit
        param_value_ind=txt.find('(', param_name_ind)
        param_unit_ind=txt[param_value_ind:].find('\t' or ')') \
            if txt[param_value_ind:].find('\t' or ')')!= -1 \
            else txt[param_value_ind:].find(')')

        param_unit=txt[param_value_ind+1:param_value_ind+param_unit_ind].strip()
        param_units[param_name]=param_unit

    return {'def_params': param_vals, 'def_units': param_units}


#####
def set_var (variation_dict):
    # get file content from baseline params and extract param section
    #variation_dict = variation_dict[0]

    for variation_per_channel in variation_dict:
        variation_dict = variation_per_channel

        channel = variation_dict['channel']
        filename = variation_dict['file'] #FILE_PER_CHANNLE[channel]

        print ('Modifying {} channel variations...'.format(channel))
        con = open_file(filename)
        indexes = get_param_section_indexes(con)
        def_info = get_def_info(con, indexes)
        def_params, def_units = def_info['def_params'], def_info['def_units']

        print ('Calculating new params ...')
        # calc new params based on the given variation dict
        DEF_PARAM = def_params
        DEF_UNITS = def_units
        new_params = calc_new_vals(variation_dict['variation'], DEF_PARAM)
        # craete the list of new params in the correct format
        new_param_list = prepare_param_section_list(new_params, DEF_UNITS)

        print ('Updating params in the file: {} ...'.format(filename))
        # update the content as an the str
        updated_con = update_content(con, new_param_list, indexes)
        # write the new file
        update_file(filename, updated_con)


    print ('Re-compiling all changes')
    #shutil.rmtree('x86_64')
    #sp.call(["nrnivmodl  mod_files/"])

    print (new_param_list)

restore_default()
#set_var(SINGLE_VAR)
