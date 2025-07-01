import clr
import os
import sys
from pspython import pspydata

# Load DLLs
scriptDir = os.path.dirname(os.path.realpath(__file__))
# This dll contains the classes in which the data is stored
clr.AddReference(scriptDir + '\\PalmSens.Core.dll')
# This dll is used to load your session file
clr.AddReference(scriptDir + '\\PalmSens.Core.Windows.dll')


# # -------------
# clr.AddReference(os.path.join(os.path.realpath(scriptDir), 'PalmSens.Core.dll'))
# clr.AddReference(os.path.join(os.path.realpath(scriptDir), 'PalmSens.Core.Simplified.WinForms.dll'))
# from PalmSens.Core.Simplified.WinForms import SimpleLoadSaveFunctions
# # -------------


# Import the static LoadSaveHelperFunctions
from PalmSens.Windows import LoadSaveHelperFunctions


def load_session_file(path, **kwargs):
    load_peak_data = kwargs.get('load_peak_data', False)
    load_eis_fits = kwargs.get('load_eis_fits', False)

    try:
        session = LoadSaveHelperFunctions.LoadSessionFile(path)
        measurements_with_curves = {}

        for m in session:
            measurements_with_curves[pspydata.convert_to_measurement(m, load_peak_data=load_peak_data, load_eis_fits=load_eis_fits)] = pspydata.convert_to_curves(m)

        return measurements_with_curves
    except:
        error = sys.exc_info()[0]
        print(error)
        return 0


def read_notes(path, n_chars=3000):
    with open(path, 'r', encoding="utf16") as myfile:
        contents = myfile.read()
    raw_txt = contents[1:n_chars].split('\\r\\n')
    notes_txt = [x for x in raw_txt if 'NOTES=' in x]
    notes_txt = notes_txt[0].replace('%20', ' ').replace('NOTES=', '').replace('%crlf', os.linesep)
    return notes_txt


def load_method_file(path):
    try:
        method = LoadSaveHelperFunctions.LoadMethod(path)
        return method
    except:
        return 0


def get_method_estimated_duration(path):
    method = load_method_file(path)
    if method == 0:
        return 0
    else:
        return method.MinimumEstimatedMeasurementDuration

