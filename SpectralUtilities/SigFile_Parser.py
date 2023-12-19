
from datetime import datetime
import json
from io import StringIO
import pandas as pd
from typing import Any

class SpectraVistaData:
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.parse_file()

    def parse_file(self) -> None:
        """
        Iterates through the lines of a .sig file creating class attributes of each label
        """
        with open(self.file_path, 'r') as file:
            spec_data_index = -1
            lines = file.readlines()[1:]
            for idx, line in enumerate(lines):
                spec_data_index = idx
                ret = self.parse_and_assign_value_attribute(line)
                if ret == 0:
                    break
            if spec_data_index > 0:
                spec_data_str = " ".join(lines[spec_data_index:])
                key, value = map(str.strip, spec_data_str.split('='))
                spec_df = pd.read_csv(StringIO(value), sep='\s+', header=None, names=["wv","1","2","refl"])
                setattr(self, key, spec_df)

    def parse_and_assign_value_attribute(self, line:str):
        """
        Convert the line into a native python type by iterating type casts and assign it as an attribute to this class.
        If all fail, we leave it as a string attribute


        line : str
            A single line from the .sig file

        """
        line = line.strip()
        key, value = map(str.strip, line.split('='))
        
        if key == "data":
            return 0
        
        # print(f"Evaluating {line}")
        value = self.try_cast(value)
            
        # print(f"Setting attribute: {key}, of type: {type(value)}, \n with value: {value}")
        setattr(self, key, value)

    def try_cast(self, value:str) -> str | list[float] | float | Any:
        """
        Brute force approach to creating a native python type from a single value of a .sig file

        value : str
            1 line from the .sig file

        Returns
        -------
            str | list[float] | float | Any: A casted value from a string
        """
        if len(value) == 0:
            return value
        elif "," in value: # Try a list of floats
            try:
                return [float(v) for v in value.split(",")]
            except:
                pass
        try: # try a single float
            return float(value)
        except:
            pass
        try: # try a dictionary
            return json.loads(value)
        except:
            pass
        try: # try a datetime stamp
            return datetime.strptime(value, "%m/%d/%Y %I:%M:%S%p")
        except:
            pass
        return value #None of them worked, just return the string type
