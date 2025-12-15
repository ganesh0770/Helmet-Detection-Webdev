export default function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-300 py-8 ">
      <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 md:grid-cols-3 gap-8">

        {/* About */}
        <div>
          <h2 className="text-xl font-semibold text-white mb-3">Helmet Detection</h2>
          <p className="text-gray-400 text-sm">
            A smart and reliable system for detecting helmet usage using AI-powered computer vision.
          </p>
        </div>

        {/* Quick Links */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Quick Links</h3>
          <ul className="space-y-2">
            <li><a href="/" className="hover:text-blue-400">Home</a></li>
            {/* <li><a href="/Live_Detection" className="hover:text-blue-400"> Live Detection</a></li> */}
            <li><a href="/about" className="hover:text-blue-400">About</a></li>
            <li><a href="/contact" className="hover:text-blue-400">Contact</a></li>
          </ul>
        </div>

        {/* Contact */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Contact</h3>
          <ul className="space-y-2 text-sm">
            <li>Email: Helmet@helmetdetection.ai</li>
            <li>Phone: xxxxxxxx</li>
            <li>Location: xxxxxx</li>
          </ul>
        </div>
      </div>

      <div className="text-center text-gray-500 text-sm mt-8 border-t border-gray-700 pt-4">
        © {new Date().getFullYear()} Helmet Detection. All rights reserved.
      </div>
    </footer>
  );
}
