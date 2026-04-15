'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Spinner } from '@/components/ui/spinner';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle, TrendingUp, BarChart3, Zap } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface PredictionResult {
  action_id: number;
  model: string;
  strategy: string;
  position: string;
  holding_days: number;
  portfolio_signal: string;
  expected_return_pct: number;
  estimated_portfolio_value: number;
  profit_loss_status: string;
}

interface NewsItem {
  date: string;
  category: string;
  headline: string;
}

interface SentimentItem {
  date: string;
  ticker: string;
  headline: string;
  sentiment_score: number;
}

interface TestVector {
  name: string;
  description: string;
  state: number[];
}

export default function TradingAgent() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [marketData, setMarketData] = useState<{ news_headlines: NewsItem[]; sentiment_analysis: SentimentItem[] } | null>(null);
  const [showMarketData, setShowMarketData] = useState(false);
  const [testScenario, setTestScenario] = useState('');
  const [selectedVector, setSelectedVector] = useState<TestVector | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const loadMarketData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/data/sample`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setMarketData(data);
      setShowMarketData(true);
    } catch (err) {
      setError(`Failed to load market data: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const runTest = async () => {
    if (!testScenario) {
      setError('Please select a test scenario');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE_URL}/test-vectors`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const vectors = await response.json();
      const vector = vectors[testScenario];

      if (!vector) {
        throw new Error('Invalid scenario selected');
      }

      setSelectedVector(vector);

      const predResponse = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: vector.state }),
      });

      if (!predResponse.ok) {
        const errorText = await predResponse.text();
        throw new Error(`API Error ${predResponse.status}: ${errorText}`);
      }

      const result = await predResponse.json();
      const pred = result.prediction || result;

      if (pred.action_id !== undefined) {
        setPrediction(pred);
      } else {
        throw new Error(result.message || 'Unknown error');
      }
    } catch (err) {
      setError(`Prediction failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.1) return 'bg-green-50 border-green-200';
    if (score < -0.1) return 'bg-red-50 border-red-200';
    return 'bg-yellow-50 border-yellow-200';
  };

  const getSentimentBadge = (score: number) => {
    if (score > 0.1) return <span className="text-green-700 font-semibold">📈 Bullish</span>;
    if (score < -0.1) return <span className="text-red-700 font-semibold">📉 Bearish</span>;
    return <span className="text-yellow-700 font-semibold">➡️ Neutral</span>;
  };

  const getPositionColor = (position: string) => {
    const lower = position.toLowerCase();
    if (lower.includes('buy') || lower.includes('long')) return 'text-green-600 bg-green-50 px-3 py-1 rounded';
    if (lower.includes('sell') || lower.includes('short')) return 'text-red-600 bg-red-50 px-3 py-1 rounded';
    return 'text-gray-600 bg-gray-50 px-3 py-1 rounded';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-slate-50">
      <div className="max-w-4xl mx-auto py-8 px-4">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-blue-600 rounded-lg">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-slate-900">RL Trading Agent</h1>
          </div>
          <p className="text-slate-600">Advanced sentiment-based trading model predictions powered by reinforcement learning</p>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert className="mb-6 border-red-200 bg-red-50">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="market" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="market">Market Data</TabsTrigger>
            <TabsTrigger value="test">Test Model</TabsTrigger>
          </TabsList>

          {/* Market Data Tab */}
          <TabsContent value="market" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Market & News Data
                </CardTitle>
                <CardDescription>Real data used in model training</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button
                  onClick={loadMarketData}
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700"
                >
                  {loading ? (
                    <>
                      <Spinner className="mr-2" />
                      Loading...
                    </>
                  ) : (
                    'Show Market Data'
                  )}
                </Button>

                {showMarketData && marketData && (
                  <div className="space-y-6">
                    {/* News Headlines */}
                    <div>
                      <h3 className="text-lg font-semibold text-slate-900 mb-3">📰 News Headlines</h3>
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {marketData.news_headlines && marketData.news_headlines.length > 0 ? (
                          marketData.news_headlines.map((item, idx) => (
                            <div
                              key={idx}
                              className="p-3 bg-white border border-slate-200 rounded-lg hover:shadow-md transition-shadow"
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex-1">
                                  <p className="font-semibold text-slate-900">{item.headline}</p>
                                  <p className="text-sm text-slate-600 mt-1">
                                    {item.date} • <span className="text-blue-600 font-medium">{item.category}</span>
                                  </p>
                                </div>
                              </div>
                            </div>
                          ))
                        ) : (
                          <p className="text-slate-600">No news data available</p>
                        )}
                      </div>
                    </div>

                    {/* Sentiment Analysis */}
                    <div>
                      <h3 className="text-lg font-semibold text-slate-900 mb-3">📊 Sentiment Analysis</h3>
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {marketData.sentiment_analysis && marketData.sentiment_analysis.length > 0 ? (
                          marketData.sentiment_analysis.map((item, idx) => {
                            const score = parseFloat(String(item.sentiment_score));
                            return (
                              <div
                                key={idx}
                                className={`p-3 border rounded-lg transition-shadow hover:shadow-md ${getSentimentColor(score)}`}
                              >
                                <div className="flex items-start justify-between gap-3">
                                  <div className="flex-1">
                                    <p className="font-semibold text-slate-900">{item.headline}</p>
                                    <p className="text-sm text-slate-600 mt-1">
                                      {item.date} • <span className="font-medium text-blue-600">{item.ticker}</span>
                                    </p>
                                  </div>
                                  <div className="text-right">
                                    {getSentimentBadge(score)}
                                    <p className="text-xs text-slate-600 mt-1">Score: {score.toFixed(3)}</p>
                                  </div>
                                </div>
                              </div>
                            );
                          })
                        ) : (
                          <p className="text-slate-600">No sentiment data available</p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Test Model Tab */}
          <TabsContent value="test" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Test Model
                </CardTitle>
                <CardDescription>Choose a test scenario to run predictions</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-900 mb-2">Select Test Scenario</label>
                  <select
                    value={testScenario}
                    onChange={(e) => setTestScenario(e.target.value)}
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">-- Select a test scenario --</option>
                    <option value="neutral_market">Neutral Market</option>
                    <option value="bullish_market">Bullish Market</option>
                    <option value="bearish_market">Bearish Market</option>
                    <option value="high_volatility">High Volatility</option>
                    <option value="positive_sentiment">Positive Sentiment Only</option>
                    <option value="negative_sentiment">Negative Sentiment Only</option>
                  </select>
                </div>

                <Button
                  onClick={runTest}
                  disabled={loading || !testScenario}
                  className="w-full bg-blue-600 hover:bg-blue-700"
                >
                  {loading ? (
                    <>
                      <Spinner className="mr-2" />
                      Running Prediction...
                    </>
                  ) : (
                    'Run Test'
                  )}
                </Button>

                {/* Selected Vector Info */}
                {selectedVector && (
                  <Card className="bg-slate-50 border-slate-200">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">{selectedVector.name}</CardTitle>
                      <CardDescription>{selectedVector.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <p className="text-xs font-medium text-slate-700">State Vector (30 features):</p>
                        <div className="bg-white p-2 rounded border border-slate-200 max-h-20 overflow-y-auto">
                          <code className="text-xs text-slate-700">{selectedVector.state.join(', ')}</code>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Prediction Results */}
                {prediction && (
                  <Card className="border-green-200 bg-green-50">
                    <CardHeader>
                      <CardTitle className="text-green-900">Prediction Results</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm font-medium text-slate-700">Model</p>
                          <p className="text-lg font-semibold text-slate-900">{prediction.model}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-700">Strategy</p>
                          <p className="text-lg font-semibold text-blue-600">{prediction.strategy}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-700">Position</p>
                          <p className={`text-lg font-semibold ${getPositionColor(prediction.position)}`}>
                            {prediction.position}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-700">Holding Days</p>
                          <p className="text-lg font-semibold text-slate-900">{prediction.holding_days}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-700">Portfolio Signal</p>
                          <p className="text-lg font-semibold text-slate-900">{prediction.portfolio_signal}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-700">Expected Return</p>
                          <p className="text-lg font-semibold text-slate-900">{prediction.expected_return_pct}%</p>
                        </div>
                        <div className="md:col-span-2">
                          <p className="text-sm font-medium text-slate-700">Estimated Portfolio Value</p>
                          <p className="text-2xl font-bold text-green-600">
                            ${prediction.estimated_portfolio_value.toLocaleString()}
                          </p>
                        </div>
                        <div className="md:col-span-2">
                          <p className="text-sm font-medium text-slate-700">Profit/Loss Status</p>
                          <p className="text-lg font-semibold text-slate-900">{prediction.profit_loss_status}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
