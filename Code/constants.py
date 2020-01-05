TOKEN_DELIM = '$'
MODE = ["english", "persian"][0]
MIN_IOU = 0.5
FILE_ADDR_PREFIX = "DataSet/phase2/"
CRAWL_START_PAGES = ["/paper/Tackling-Climate-Change-with-Machine-Learning-Rolnick-Donti/998039a4876edc440e0cabb0bc42239b0eb29644",
                     "/paper/Sublinear-Algorithms-for-(%CE%94%2B-1)-Vertex-Coloring-Assadi-Chen/eb4e84b8a65a21efa904b6c30ed9555278077dd3",
                     "/paper/Processing-Data-Where-It-Makes-Sense%3A-Enabling-Mutlu-Ghose/4f17bd15a6f86730ac2207167ccf36ec9e6c2391"]
CRAWLER_PREFIX = "https://www.semanticscholar.org"
DESIRED_PAGES_NUM = 8000
CRAWLER_OUTPUT_JSON = "DataSet/phase3/crawler.json"
CRAWLER_DUMP_FILE_ADDR = "DataSet/phase3/crawler_dump.py"
PAGE_RANK_ALPHA = 0.2