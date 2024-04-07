from typing import List
from lime.lime_text import LimeTextExplainer
from pathlib import Path

class UrlExplainer(LimeTextExplainer):

    def __init__(self,
                 save_dir,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False):
        super().__init__(kernel_width, kernel, verbose, class_names, feature_selection, split_expression, bow, mask_string, random_state, char_level)
        self.last_instances_explained = []
        self.save_dir = Path(save_dir)

        assert self.save_dir.exists(), f"Directory {self.save_dir} does not exist"

    def explain_list(self, urls: List[str], func) -> List[str]:
        self.last_instances_explained = []
        
        for url in urls:
            print(url)
            exp = self.explain_instance("test.com", func)
            self.last_instances_explained.append(exp)
        
        return self.last_instances_explained
    
    def show_last_explanations(self, show_in_notebook=False):
        
        for i, instance in enumerate(self.last_instances_explained):
            print(f"Instance {i}")
            print(instance)
            print("\n")

            if not show_in_notebook:
                fig = instance.as_pyplot_figure()
                fig.savefig(f"{self.save_dir}/exp_instance_{i}.png")
                instance.save_to_file(f"{self.save_dir}/exp_instance_{i}.html")
            else:
                instance.show_in_notebook(text=True)
    