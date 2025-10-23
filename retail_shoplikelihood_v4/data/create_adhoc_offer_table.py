from app.services.sharepoint_manager import Sharepoint
from app.util.timer import Timer
from config import params, settings
import logging


logger = logging.getLogger(__name__)


class CreateAdHocOfferInfoFromCSV:
    def __init__(self):
        self._base_url = params.SHAREPOINT_BASE_URL
        self._file_loc = params.ADHOC_OFFER_CSV_FILE_LOC
        self.excel_filename = 'ad_hoc_offers_master_tracker.xlsx'
        self.destination_dir = settings.PROJECT_DIR + '/tmp_files/'
        self.destination_filename = self.destination_dir + self.excel_filename
        self.sharepoint_manager = Sharepoint(sharepoint_base_url=self._base_url, folder_in_sharepoint=self._file_loc)
        self.week_identifier = params.WEEK_NUMBER
    

    def download_excel_from_sharepoint(self):   
        logger.info(f"Downloading `{self.excel_filename}` from SharePoint to `{self.destination_filename}`")    
        self.sharepoint_manager.download(source_filename=self.excel_filename, destination_filename=self.destination_filename)

   
    @Timer
    def process(self):
        self.download_excel_from_sharepoint()
        

if __name__ == "__main__":
    create_ad_hoc_offer_info_from_csv = CreateAdHocOfferInfoFromCSV()
    create_ad_hoc_offer_info_from_csv.process()
