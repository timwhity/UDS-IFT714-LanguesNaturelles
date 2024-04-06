from typing import List
import re
from numpy import array, int32

class FeatureExtractor:
    """
    Generate features from the url
    """
    def __init__(self) -> None:
        pass

    def extract(self, url: str) -> array:
        """
        Input : string of the url
        Output : numpy array of features extracted from the url
        """
        if url.startswith('https://'):
            https_mode = 1
        elif url.startswith('http://'):
            https_mode = 0
        else:
            https_mode = 2
        
        if len(url.split('://')) > 1:
            url_without_protocol = url.split('://')[1]                  # remove http:// or https://     Ex : www.google.com/abc/def/test.py?query=123
        else:
            url_without_protocol = url
        url_without_protocol_query = url_without_protocol.split('?')[0] # remove query string            Ex : www.google.com/abc/def/test.py
        domains = url_without_protocol_query.split('/')[0]              # remove path                    Ex : www.google.com
        top_level_domain = domains.split('.')[-1]                       # get top level domain           Ex : com
        directory = '/'.join(url_without_protocol_query.split('/')[1:-1])# get directory path            Ex : abc/def
        file = url_without_protocol_query.split('/')[-1]                # get file name                  Ex : test.py
        query = url_without_protocol.split('?')[1] if '?' in url_without_protocol else '' # get query string           Ex : query=123

        features = array([
            # url,
            url.count('.'),
            url.count('-'),
            url.count('_'),
            url.count('/'),
            url.count('?'),
            url.count('='),
            url.count('@'),
            url.count('&'),
            url.count('!'),
            url.count(' '),
            url.count('~'),
            url.count(','),
            url.count('+'),
            url.count('*'),
            url.count('#'),
            url.count('$'),
            url.count('%'),
            url.count('%'),
            len(top_level_domain),
            len(url),
            domains.count('.'),
            domains.count('-'),
            domains.count('_'),
            domains.count('/'),
            domains.count('?'),
            domains.count('='),
            domains.count('@'),
            domains.count('&'),
            domains.count('!'),
            domains.count(' '),
            domains.count('~'),
            domains.count(','),
            domains.count('+'),
            domains.count('*'),
            domains.count('#'),
            domains.count('$'),
            domains.count('%'),
            self.count_vowels(domains),
            len(domains),
            int(self.is_domain_in_ip_format(domains)),
            int(domains.find('server') != -1 or domains.find('client') != -1),
            directory.count('.'),
            directory.count('-'),
            directory.count('_'),
            directory.count('/'),
            directory.count('?'),
            directory.count('='),
            directory.count('@'),
            directory.count('&'),
            directory.count('!'),
            directory.count(' '),
            directory.count('~'),
            directory.count(','),
            directory.count('+'),
            directory.count('*'),
            directory.count('#'),
            directory.count('$'),
            directory.count('%'),
            len(directory),
            file.count('.'),
            file.count('-'),
            file.count('_'),
            file.count('/'),
            file.count('?'),
            file.count('='),
            file.count('@'),
            file.count('&'),
            file.count('!'),
            file.count(' '),
            file.count('~'),
            file.count(','),
            file.count('+'),
            file.count('*'),
            file.count('#'),
            file.count('$'),
            file.count('%'),
            len(file),
            query.count('.'),
            query.count('-'),
            query.count('_'),
            query.count('/'),
            query.count('?'),
            query.count('='),
            query.count('@'),
            query.count('&'),
            query.count('!'),
            query.count(' '),
            query.count('~'),
            query.count(','),
            query.count('+'),
            query.count('*'),
            query.count('#'),
            query.count('$'),
            query.count('%'),
            len(query),
            query.count('=') + 1 if query else 0,
            int(query.find(top_level_domain) != -1),
            https_mode,
            int(re.search(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", url) != None),
        ], dtype=int32)
        return features

    def extract_batch(self, urls: List[str]) -> array:
        """
        Input : list of urls
        Output : numpy array of features extracted from the urls, size = (len(urls), nb_features)
        """
        return array([self.extract(url) for url in urls])

    def count_vowels(self, string: str) -> int:
        return sum(1 for char in string if char in 'aeiouyAEIOUY')

    def is_domain_in_ip_format(self, domain: str) -> bool:
        return domain.replace('.', '').isnumeric()
    


def main():
    dataset = "combined_dataset_12/all.csv"
    import pandas as pd
    df = pd.read_csv(dataset)
    urls = df['url'].values

    extractor = FeatureExtractor()
    features = extractor.extract_batch(urls)
    # features = extractor.extract("www.google.com/abc/def/test.py?query=123")
    # print(features)
    print(features.shape)
    # print(features)
    
    # for url, feature in zip(urls, features):
    #     print(url)
    #     print(feature)
    #     print()

if __name__ == '__main__':
    main()