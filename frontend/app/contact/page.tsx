"use client";
import { useState } from "react";


export default function ContactPage() {
  const [respnse, setrespnse] = useState(false)
  const handle_submit = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    const res = await fetch("http://127.0.0.1:8000/contact_data", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    console.log("respond from backend", data);
    setrespnse(data)
  };
  const handleChange=()=>{
    setrespnse(false)

  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-200 py-20 px-6 ">
      <div className="max-w-5xl mx-auto space-y-16">

        {/* Page Title */}
        <div className="text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-white">
            Contact Us
          </h1>
          <p className="mt-4 text-lg text-gray-400">
            Have questions or need support? We're here to help.
          </p>
        </div>

        {/* Contact Section */}
        <div className="grid md:grid-cols-2 gap-12">

          {/* Contact Form */}
          <form
            className="bg-gray-900 border border-gray-800 p-8 rounded-xl space-y-6 shadow-lg" id="myform" onSubmit={handle_submit}
          >
            <div>
              <label className="block text-gray-300 mb-2">Full Name</label>
              <input
              onChange={handleChange}
                type="text"
                name="name"
                id="name"
                className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="Enter your name"
              />
            </div>

            <div>
              <label className="block text-gray-300 mb-2">Email Address</label>
              <input
              onChange={handleChange}
                type="email"
                name="email"
                id="email"
                className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="your@email.com"
              />
            </div>

            <div>
              <label className="block text-gray-300 mb-2">Message</label>
              <textarea
              onChange={handleChange}
                name="text_box"
                id="text_box"
                rows={5}
                className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="Write your message..."
              ></textarea>
            </div>

            <button
              type="submit"
              id="submit_button"
              className="w-full py-3 bg-blue-600 hover:bg-blue-700 transition rounded-lg font-semibold"
            >
              Send Message
            </button>

            {respnse  && (

              <div className=" bg-blue-500 text-center font-bold py-1 rounded-4xl w-full  ">
                Saved
              </div>
            )}

          </form>

          {/* Contact Details */}
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Get in Touch</h2>
              <p className="text-gray-400 leading-relaxed">
                Whether you have technical questions, business inquiries, or
                need support with helmet detection, feel free to contact us.
              </p>
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-semibold text-white">📧 Email</h3>
                <p className="text-gray-400">Helmet@helmetdetection.ai</p>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-white">📞 Phone</h3>
                <p className="text-gray-400">XXXXXXX</p>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-white">📍 Address</h3>
                <p className="text-gray-400">
                  XXXXXXX
                </p>
              </div>
            </div>

          </div>
        </div>


        {/* Map or Additional Info */}
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-xl text-center">
          <p className="text-gray-400">
            You can also reach us through our social channels or schedule a demo anytime.
          </p>
        </div>

      </div>
    </div>
  );
}





