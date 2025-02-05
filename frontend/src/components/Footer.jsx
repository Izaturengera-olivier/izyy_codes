import { Facebook, Twitter, Instagram, Mail, Phone } from "lucide-react";

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-gray-300 py-10">
      <div className="max-w-6xl mx-auto px-6 grid md:grid-cols-3 gap-8">
        {/* About Section */}
        <div>
          <h2 className="text-2xl font-bold text-white">MyBrand</h2>
          <p className="mt-3 text-gray-400">
            We provide insightful articles and valuable resources to help you stay informed.
          </p>
        </div>

        {/* Quick Links */}
        <div>
          <h3 className="text-xl font-semibold text-white">Quick Links</h3>
          <ul className="mt-3 space-y-2">
            <li><a href="/" className="hover:text-blue-400 transition">Home</a></li>
            <li><a href="/about" className="hover:text-blue-400 transition">About Us</a></li>
            <li><a href="/contact" className="hover:text-blue-400 transition">Contact</a></li>
            <li><a href="/blog" className="hover:text-blue-400 transition">Blog</a></li>
          </ul>
        </div>

        {/* Contact Info */}
        <div>
          <h3 className="text-xl font-semibold text-white">Contact Us</h3>
          <ul className="mt-3 space-y-2">
            <li className="flex items-center gap-2">
              <Mail size={18} /> izzyy720@gmail.com
            </li>
            <li className="flex items-center gap-2">
              <Phone size={18} /> +250 780463588
            </li>
          </ul>
          {/* Social Icons */}
          <div className="flex gap-4 mt-4">
            <a href="#" className="text-gray-400 hover:text-blue-500 transition"><Facebook size={24} /></a>
            <a href="#" className="text-gray-400 hover:text-blue-500 transition"><Twitter size={24} /></a>
            <a href="#" className="text-gray-400 hover:text-pink-500 transition"><Instagram size={24} /></a>
          </div>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="border-t border-gray-700 mt-8 pt-6 text-center text-gray-400">
        © {new Date().getFullYear()} MyBrand. All rights reserved.
      </div>
    </footer>
  );
};

export default Footer;
