import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import StockDetail from './pages/StockDetail'
import Portfolio from './pages/Portfolio'
import Watchlist from './pages/Watchlist'
import Compare from './pages/Compare'
import AIPredictor from './pages/AIPredictor'
import News from './pages/News'
import PrivacyPolicy from './pages/PrivacyPolicy'
import TermsAndConditions from './pages/TermsAndConditions'
import Contact from './pages/Contact'
import Login from './pages/Login'
import Register from './pages/Register'
import NotFound from './pages/NotFound'
import RiskDisclosure from './pages/RiskDisclosure'
import Trademark from './pages/Trademark'
import CookiePolicy from './pages/CookiePolicy'

function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Navbar />
        <main className="flex-grow container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/stock/:symbol" element={<StockDetail />} />
            <Route path="/ai-predictor" element={<AIPredictor />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/watchlist" element={<Watchlist />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/news" element={<News />} />
            <Route path="/privacy-policy" element={<PrivacyPolicy />} />
            <Route path="/terms-and-conditions" element={<TermsAndConditions />} />
            <Route path="/risk-disclosure" element={<RiskDisclosure />} />
            <Route path="/trademark" element={<Trademark />} />
            <Route path="/cookie-policy" element={<CookiePolicy />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </AuthProvider>
  )
}

export default App
