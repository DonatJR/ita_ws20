import logging
import logging.config

import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from ..items import PdfsItem

# Init logger
logging.config.fileConfig(fname="logging.conf")
slogger = logging.getLogger("stats")


class JmlrSpider(CrawlSpider):
    name = "jmlr"
    allowed_domains = ["jmlr.csail.mit.edu"]
    start_urls = ["https://jmlr.csail.mit.edu/"]

    rules = (Rule(LinkExtractor(allow=r"papers/"), callback="parse_item", follow=True),)

    def parse_item(self, response):
        file_url = response.css('a[target ="_blank"]::attr(href)').extract()

        if file_url == []:
            file_url = response.css('a[id ="pdf"]::attr(href)').extract()

        if file_url != []:
            if not file_url[0].startswith("http"):
                file_url = "https://jmlr.csail.mit.edu/" + file_url[0]
            else:
                file_url = file_url[0]

            slogger.info(file_url)

            item = PdfsItem()
            item["file_urls"] = [file_url]
            yield item
