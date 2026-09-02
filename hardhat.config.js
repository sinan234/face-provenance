// Minimal Hardhat config so `npx -y hardhat node` boots a local Ethereum
// dev chain on http://127.0.0.1:8545 (chain id 31337) with 20 funded
// accounts. Account #0 uses the well-known dev key already referenced in
// .env.example — LOCAL DEV ONLY, never use it on a real network.
module.exports = {
  defaultNetwork: "localhost",
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
    },
  },
};
