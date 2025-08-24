import React from 'react'
import { Link } from 'react-router-dom'
import { Mail, Phone, MapPin, Clock, ArrowLeft, MessageSquare } from 'lucide-react'

const Contact = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center mb-4">
          <MessageSquare className="h-12 w-12 text-blue-600" />
        </div>
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Contact Us</h1>
        <p className="text-gray-600">Get in touch with our team</p>
      </div>

      {/* Back Button */}
      <Link 
        to="/" 
        className="inline-flex items-center text-blue-600 hover:text-blue-800 transition-colors"
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Home
      </Link>

      {/* Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Contact Information */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">Get In Touch</h2>
          
          <div className="space-y-6">
            <div className="flex items-start space-x-4">
              <Mail className="h-6 w-6 text-blue-600 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-medium text-gray-900">Email</h3>
                <p className="text-gray-600 mt-1">
                  <a href="mailto:support@stockanalyzer.com" className="hover:text-blue-600 transition-colors">
                    support@stockanalyzer.com
                  </a>
                </p>
                <p className="text-sm text-gray-500 mt-1">For general inquiries and support</p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <Phone className="h-6 w-6 text-blue-600 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-medium text-gray-900">Phone</h3>
                <p className="text-gray-600 mt-1">
                  <a href="tel:+1-800-786-2501" className="hover:text-blue-600 transition-colors">
                    1-800-STOCK-01
                  </a>
                </p>
                <p className="text-sm text-gray-500 mt-1">Monday - Friday, 9 AM - 6 PM EST</p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <MapPin className="h-6 w-6 text-blue-600 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-medium text-gray-900">Address</h3>
                <p className="text-gray-600 mt-1">
                  123 Financial District<br />
                  New York, NY 10001<br />
                  United States
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <Clock className="h-6 w-6 text-blue-600 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-medium text-gray-900">Business Hours</h3>
                <div className="text-gray-600 mt-1 space-y-1">
                  <p>Monday - Friday: 9:00 AM - 6:00 PM EST</p>
                  <p>Saturday: 10:00 AM - 2:00 PM EST</p>
                  <p>Sunday: Closed</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Contact Form */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">Send us a Message</h2>
          
          <form className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="firstName" className="block text-sm font-medium text-gray-700 mb-2">
                  First Name
                </label>
                <input
                  type="text"
                  id="firstName"
                  name="firstName"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  placeholder="John"
                />
              </div>
              <div>
                <label htmlFor="lastName" className="block text-sm font-medium text-gray-700 mb-2">
                  Last Name
                </label>
                <input
                  type="text"
                  id="lastName"
                  name="lastName"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  placeholder="Doe"
                />
              </div>
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="john.doe@example.com"
              />
            </div>

            <div>
              <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
                Subject
              </label>
              <select
                id="subject"
                name="subject"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              >
                <option value="">Select a subject</option>
                <option value="support">Technical Support</option>
                <option value="feature">Feature Request</option>
                <option value="bug">Bug Report</option>
                <option value="account">Account Issues</option>
                <option value="partnership">Partnership Inquiry</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div>
              <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-2">
                Message
              </label>
              <textarea
                id="message"
                name="message"
                rows={6}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors resize-none"
                placeholder="Please describe your inquiry or feedback..."
              ></textarea>
            </div>

            <button
              type="submit"
              className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              Send Message
            </button>
          </form>

          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> We typically respond to inquiries within 24 hours during business days. 
              For urgent technical issues, please call our support line directly.
            </p>
          </div>
        </div>
      </div>

      {/* FAQ Section */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-semibold text-gray-900 mb-6">Frequently Asked Questions</h2>
        
        <div className="space-y-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">How accurate are the stock predictions?</h3>
            <p className="text-gray-600 text-sm">
              Our AI models provide predictions based on historical data and market patterns. However, stock markets 
              are inherently unpredictable, and all predictions should be used for informational purposes only.
            </p>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-2">Is my portfolio data secure?</h3>
            <p className="text-gray-600 text-sm">
              Yes, we use industry-standard encryption and security measures to protect your data. Your portfolio 
              information is stored securely and is never shared with third parties.
            </p>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-2">Can I export my portfolio data?</h3>
            <p className="text-gray-600 text-sm">
              Yes, you can export your portfolio data in various formats including CSV and PDF through your 
              account settings.
            </p>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-2">How often is the stock data updated?</h3>
            <p className="text-gray-600 text-sm">
              Our stock data is updated in real-time during market hours. After-hours data may have slight delays.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Contact
