import os
from torch.utils.data import Dataset, DataLoader
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_test_case(model, tokenizer, input_method, device):
    # Tokenize the input method
    tokenized_input = tokenizer(input_method, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
    tokenized_input = {key: val.to(device) for key, val in tokenized_input.items()}

    # Generate a test case using the model
    with torch.no_grad():
        model.eval()
        output = model.generate(
            **tokenized_input,
            max_length=1024,
            # num_return_sequences=1,
            # do_sample=True,
            # num_beams=5,
            # temperature=1.0,
            # no_repeat_ngram_size=2
        )

    # Decode the generated test case
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded_output

# Load the fine-tuned model and tokenizer
# model_path = "saved_models/local_raw"
model_path = "Minata/plbart-base_finetuned_ut_generator_70000_method2test"
# model_path = "output/checkpoints1e-5"
tokenizer = PLBartTokenizer.from_pretrained(model_path)
model = PLBartForConditionalGeneration.from_pretrained(model_path).to(device)

input_method = """public Float getAccountPortfolioValue(String accountId) throws AccountNotFoundException, ResourceNotFoundException { List<Asset> userAssets = this.listAssets(accountId); float total = 0f; for (Asset asset : userAssets) { if (asset.getTradableType().equals("stock")) { Stock stock = stockService.getStockById(asset.getTradableId()); total += stock.getPrice() * asset.getQuantity(); } else if (asset.getTradableType().equals("cryptocurrency")) { Cryptocurrency crypto = cryptocurrencyService .getCryptocurrencyById(asset.getTradableId()); total += crypto.getPrice() * asset.getQuantity(); } else if (asset.getTradableType().equals("nft")) { NFT nft = nftService.getNFTById(asset.getTradableId()); total += nft.getPrice() * asset.getQuantity(); } else { throw new ResourceNotFoundException("pnl functions for asset type " + asset.getTradableType() + " not implemented"); } } return total; }"""
target_method = """@Test public void getPortfolioValue() throws AccountNotFoundException, ResourceNotFoundException { doReturn(stock).when(mockStockService).getStockById(stock.getStockId()); doReturn(nft).when(mockNFTService).getNFTById(nft.getNftId()); doReturn(crypto).when(mockCryptoService).getCryptocurrencyById(crypto.getCryptocurrencyId()); doReturn(assets).when(mockAssetRepository).findAllAssetsByAccountId(accountId); Float portfolioValue = assetService.getAccountPortfolioValue(accountId); assertEquals(portfolioValue, portfolioValueTruth); }"""

#input_method = """RosetteAbstractProcessor extends AbstractProcessor { @Override public IngestDocument execute(IngestDocument ingestDocument) throws Exception { if (ingestDocument.hasField(targetField)) { throw new ElasticsearchException("Document already contains data in target field for this ingest " + "processor: " + type); } if (!ingestDocument.hasField(inputField)) { return ingestDocument; } String inputText = ingestDocument.getFieldValue(inputField, String.class); if (Strings.isNullOrEmpty(inputText)) { return ingestDocument; } SecurityManager sm = System.getSecurityManager(); if (sm != null) { sm.checkPermission(new SpecialPermission()); } processDocument(inputText, ingestDocument); return ingestDocument; } RosetteAbstractProcessor(RosetteApiWrapper rosAPI, String tag, String description, String processorType,\n String inputField, String targetField); }"""

# Generate and print the test case
prediction = generate_test_case(model, tokenizer, input_method, device)
print(prediction)

bleu_score = 0
smoothing = SmoothingFunction()
bleu_score += sentence_bleu(
    [target_method.split()],
    prediction.split(),
    smoothing_function=smoothing.method1,
)
print(f"BLEU score: {bleu_score}")