import React from 'react'
import { Link } from 'react-router-dom'
import { 
  Facebook, 
  Linkedin, 
  Instagram, 
  Twitter, 
  Youtube,
  Mail,
  Phone,
  MapPin,
  TrendingUp
} from 'lucide-react'

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Company Info */}
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <TrendingUp className="h-8 w-8 text-blue-400" />
              <span className="text-xl font-bold">Stock Analyzer</span>
            </div>
            <p className="text-gray-400 mb-4">
              Your comprehensive platform for stock market analysis and portfolio management. 
              Make informed investment decisions with real-time data and AI-powered predictions.
            </p>
            <div className="flex space-x-4">
              <a 
                href="https://facebook.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-400 transition-colors"
                aria-label="Facebook"
              >
                <Facebook className="h-5 w-5" />
              </a>
              <a 
                href="https://linkedin.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-400 transition-colors"
                aria-label="LinkedIn"
              >
                <Linkedin className="h-5 w-5" />
              </a>
              <a 
                href="https://instagram.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-400 transition-colors"
                aria-label="Instagram"
              >
                <Instagram className="h-5 w-5" />
              </a>
              <a 
                href="https://twitter.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-400 transition-colors"
                aria-label="Twitter"
              >
                <Twitter className="h-5 w-5" />
              </a>
              <a 
                href="https://youtube.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-400 transition-colors"
                aria-label="YouTube"
              >
                <Youtube className="h-5 w-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-white mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link to="/" className="text-gray-400 hover:text-white transition-colors">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/watchlist" className="text-gray-400 hover:text-white transition-colors">
                  Watchlist
                </Link>
              </li>
              <li>
                <Link to="/portfolio" className="text-gray-400 hover:text-white transition-colors">
                  Portfolio
                </Link>
              </li>
              <li>
                <Link to="/compare" className="text-gray-400 hover:text-white transition-colors">
                  Stock Comparison
                </Link>
              </li>
              <li>
                <Link to="/news" className="text-gray-400 hover:text-white transition-colors">
                  Market News
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="font-semibold text-white mb-4">Legal</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link to="/privacy-policy" className="text-gray-400 hover:text-white transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link to="/terms-and-conditions" className="text-gray-400 hover:text-white transition-colors">
                  Terms & Conditions
                </Link>
              </li>
              <li>
                <Link to="/risk-disclosure" className="text-gray-400 hover:text-white transition-colors">
                  Risk Disclosure
                </Link>
              </li>
              <li>
                <Link to="/trademark" className="text-gray-400 hover:text-white transition-colors">
                  Trademark
                </Link>
              </li>
              <li>
                <Link to="/cookie-policy" className="text-gray-400 hover:text-white transition-colors">
                  Cookie Policy
                </Link>
              </li>
            </ul>
          </div>

          {/* Contact & Support */}
          <div>
            <h4 className="font-semibold text-white mb-4">Contact & Support</h4>
            <div className="space-y-3 text-sm">
              <div className="flex items-center space-x-3 text-gray-400">
                <Mail className="h-4 w-4 flex-shrink-0" />
                <a 
                  href="mailto:support@stockanalyzer.com" 
                  className="hover:text-white transition-colors"
                >
                  support@stockanalyzer.com
                </a>
              </div>
              <div className="flex items-center space-x-3 text-gray-400">
                <Phone className="h-4 w-4 flex-shrink-0" />
                <a 
                  href="tel:+1-800-STOCK-01" 
                  className="hover:text-white transition-colors"
                >
                  1-800-STOCK-01
                </a>
              </div>
              <div className="flex items-start space-x-3 text-gray-400">
                <MapPin className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <span>
                  123 Financial District<br />
                  New York, NY 10001<br />
                  United States
                </span>
              </div>
            </div>
            
            <div className="mt-4">
              <Link 
                to="/contact" 
                className="text-blue-400 hover:text-blue-300 transition-colors text-sm"
              >
                Contact Us →
              </Link>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-800 mt-8 pt-8">
          <div className="flex flex-col lg:flex-row justify-between items-center gap-4">
            <div className="text-sm text-gray-400">
              <p>© 2025 Stock Analyzer. All rights reserved.</p>
            </div>
            
            <div className="text-sm text-gray-400 text-center lg:text-right">
              <p>
                Data provided by{' '}
                <a 
                  href="https://finance.yahoo.com" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 transition-colors"
                >
                  Yahoo Finance
                </a>
                {' • '}
                Real-time and delayed market data
              </p>
            </div>
          </div>
          
          <div className="mt-4 text-xs text-gray-500 text-center">
            <p>
              <strong>Disclaimer:</strong> This platform is for informational purposes only and does not constitute financial advice. 
              All investments carry risk and past performance does not guarantee future results. 
              Please consult with a qualified financial advisor before making investment decisions.
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
