
class Plotter:
    def __init__(self, figsize=[10, 10], fontsize=12, fontfamily='sans-serif',
                 fontweight='normal', dpi: int = 200, max_col: int = 1):
        self.figsize = figsize
        self.fontsize = fontsize
        self.fontfamily = fontfamily
        self.fontweight = fontweight
        self.dpi = dpi
        self.max_col
        self.metric_dic = {}

    def fit(self):
        """
        Fits the dictionnary of metrics. At the end of every
        """



    # Group files by metric name
    filenames_dic = nested_default_dict(4, str)
    for path in all_files:
        root = path.parent
        metric = path.stem
        with open(root / 'config.json') as f:
            config = json.load(f)
        fixed_key = tuple([config[key] for key in simu_params])
        reduce_key = config[reduce_by]
        filenames_dic[metric][fixed_key][reduce_key]['log_freq'] = config['log_freq']
        filenames_dic[metric][fixed_key][reduce_key]['path'] = path